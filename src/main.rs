use std::collections::HashMap;
use dashmap::DashMap;
use std::hash::Hasher;
use unicode_segmentation::UnicodeSegmentation;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::collections::VecDeque;
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Error, Result};
use rayon::prelude::*;
use serde_json;
use serde_json::{Value, json};
use clap::{Parser, Subcommand};


use mj_io::{
    build_pbar, expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf, get_output_filename
};

/*
Maybe I keep record of which reference docs and which byte indices?
*/
/*===============================================
=                      COMMAND BLOCK            =
===============================================*/




#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct ArgParser {
    #[clap(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    #[clap(arg_required_else_help = true)]
    Decontam {
        #[arg(required = true, long)]
        reference_dir: PathBuf,

        #[arg(required=true, long)]
        reference_key: String,

        #[arg(required=true, long)]
        train_text_key: String,

        #[arg(required = true, long)]
        input_dir: PathBuf,

        #[arg(required = true, long)]
        output_dir: PathBuf,

        #[arg[required = true, long]]
        ngrams: usize,

        #[arg[long, default_value_t = false]]
        keep_only_clean: bool,
    },

}


/*=========================================================
=                     UTILITIES                           =
=========================================================*/
fn normalize_text(text: &String) -> String {
    // Remove all punctuation, downcase the text, and replace newlines/tabs with spaces
    text.chars()
        .filter_map(|c| {
            if c.is_ascii_punctuation() {
                None // This skips the character entirely
            } else if c == '\n' || c == '\t' {
                Some(' ') // Replace newlines/tabs with spaces
            } else {
                Some(c.to_lowercase().next().unwrap_or(c))
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

fn get_ngrams_idxs(text_str: &str, ngram_size: usize) -> Result<Vec<(usize, usize)>, Error> {
    // Gets a list of <idx_start, ngram_hash> where idx_start is the index into the string
    let mut ngram: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut idx_starts: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut output: Vec<(usize, usize)> = Vec::new();

    text_str.split_word_bound_indices().for_each(|(i,w)| {
        if !w.trim().is_empty() {        
            let word_hash = hash_object::<str>(w);
            idx_starts.push_back(i);
            ngram.push_back(word_hash);
            if ngram.len() >= ngram_size {
                let ngram_hash = hash_object::<VecDeque<usize>>(&ngram);
                let first_idx = idx_starts.pop_front().unwrap_or_default();
                ngram.pop_front();
                output.push((first_idx, ngram_hash));
            }
        }
    });
    Ok(output)
}

fn hash_object<T: Hash + ?Sized>(obj: &T) -> usize {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish() as usize    
}


/*=========================================================
=                     DECONTAMINATION BLOCK               =
=========================================================*/

fn dencontam(reference_dir: &PathBuf, reference_key: &String, train_text_key: &String, input_dir: &PathBuf, output_dir: &PathBuf, ngram_size: usize, keep_only_clean: bool) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting decontam!");

    // First build the reference pool
    let start_ref = Instant::now();
    let reference_ngrams = process_eval_set(reference_dir, reference_key, ngram_size).unwrap();
    println!("Built reference lookup in {:?} secs", start_ref.elapsed().as_secs());


    // Then loop through and collect to see if anything matches
    // Remove nothing, but do annotate with matches
    println!("Starting decontamination");

    let total_docs = AtomicUsize::new(0);
    let contam_docs = AtomicUsize::new(0);
    let input_paths = expand_dirs(vec![input_dir.clone()], None).unwrap();
    let pbar = build_pbar(input_paths.len(), "input paths");
    input_paths.into_par_iter().for_each(|p| {
        let output_path = get_output_filename(&p, input_dir, output_dir).unwrap();
        let (path_docs, path_contam_docs) = decontam_path(train_text_key, &p, &output_path, &reference_ngrams, ngram_size, keep_only_clean).unwrap();
        total_docs.fetch_add(path_docs, Ordering::SeqCst);
        contam_docs.fetch_add(path_contam_docs, Ordering::SeqCst);
        pbar.inc(1);
    });


    let total_docs = total_docs.into_inner();
    let contam_docs = contam_docs.into_inner();
    println!("Finished decontam in {:?} seconds", start_main.elapsed().as_secs());
    println!("Saw {:?} contam out of {:?} docs for a contam rate of {:.2}%", 
             contam_docs, total_docs, (contam_docs as f32 / total_docs as f32) * 100.0);
    Ok(())
}


fn process_eval_set(input_dir: &PathBuf, text_key: &str, ngram_size: usize) -> Result<DashMap<usize, Vec<(usize, usize, usize)>>, Error> {
    // Load all elements in the eval set 
    // Process each file, and annotate it with the normalized-text (and save these in the same locations)
    // Also creates the dashmap of ngram-hash -> [(file_id, line_num, ngram_start_idx),...]

    let mut input_paths = expand_dirs(vec![input_dir.clone()], None).unwrap();
    input_paths.sort_by(|a, b| {
        let a_name = a.file_name().unwrap_or_default();
        let b_name = b.file_name().unwrap_or_default();
        a_name.cmp(&b_name)
    });
    let norm_text_key = text_key.to_owned() + "_normalized";
    let eval_hashes : DashMap<usize, Vec<(usize, usize, usize)>> = DashMap::new();
    let pbar = build_pbar(input_paths.len(), "Eval paths");
    input_paths.into_iter().enumerate().for_each(|(path_id, path)| {
        let contents = read_pathbuf_to_mem(&path).unwrap();
        let mut output_bytes : Vec<u8> = Vec::new();
        for (line_num, line) in contents.lines().enumerate() {
            let line = line.unwrap();
            let mut line_json: Value = serde_json::from_str(&line).unwrap();
            let base_text = line_json.get(text_key).unwrap().as_str().unwrap().to_string();
            let norm_text = normalize_text(&base_text);
            line_json[norm_text_key.clone()] = json!(norm_text);
            let ngram_idxs = get_ngrams_idxs(&norm_text, ngram_size).unwrap();
            for (idx, ngram_hash) in ngram_idxs {
                eval_hashes.entry(ngram_hash).or_default().push((path_id, line_num, idx));
            }            
            output_bytes.extend(serde_json::to_vec(&line_json).unwrap());
            output_bytes.push(b'\n');
        }
        write_mem_to_pathbuf(&output_bytes, &path).unwrap();
        pbar.inc(1);
    });


    Ok(eval_hashes)

}   

fn decontam_path(train_text_key: &String, input_path: &PathBuf, output_path: &PathBuf, reference_ngrams: &DashMap<usize, Vec<(usize, usize, usize)>>, 
                 ngram_size: usize, keep_only_clean: bool) -> Result<(usize, usize), Error> {
    // Annotates path with norm_text, and also the set of contams
    // contam: {text_idx: [(file_id, line_num, ngram_start_idx), ...]}

    let contents = read_pathbuf_to_mem(input_path).unwrap();
    let mut total_lines = 0;
    let mut contam_lines = 0;
    let mut output_bytes: Vec<u8> = Vec::new();
    let norm_text_key = train_text_key.clone() + "_normalized";
    for line in contents.lines() {
        total_lines += 1;
        let line = line.unwrap();
        let mut line_json: Value = serde_json::from_str(&line).unwrap();
        let text = line_json.get(&train_text_key).unwrap().as_str().unwrap().to_string();
        let norm_text = normalize_text(&text);
        line_json[&norm_text_key] = json!(norm_text);
        let ngrams = get_ngrams_idxs(&norm_text, ngram_size).unwrap();
        let mut decon: HashMap<usize, Vec<(usize, usize, usize)>> = HashMap::new();
        for (idx, ngram) in ngrams {
            if reference_ngrams.contains_key(&ngram) {
                decon.insert(idx, reference_ngrams.get(&ngram).unwrap().to_vec());
            }
        }
        if decon.len() > 0 {
            line_json["contamination"] = json!(decon);
            contam_lines += 1;
        }

        if !keep_only_clean || (keep_only_clean && decon.len() == 0) {
            output_bytes.extend(serde_json::to_vec(&line_json).unwrap());
            output_bytes.push(b'\n');
        }
    }

    write_mem_to_pathbuf(&output_bytes, output_path).unwrap();
    Ok((total_lines, contam_lines))
}


/*===========================================
=                    MAIN BLOCK             =
===========================================*/



#[allow(unreachable_patterns)]
fn main() {
    let args = ArgParser::parse();
    let result = match &args.command {
        Commands::Decontam {
            reference_dir,
            reference_key,
            train_text_key,
            input_dir,
            output_dir,
            ngrams,   
            keep_only_clean     
        } => dencontam(reference_dir, reference_key, train_text_key, input_dir, output_dir, *ngrams, *keep_only_clean),
    };
    result.unwrap();
}


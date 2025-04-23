use std::hash::Hasher;
use unicode_segmentation::UnicodeSegmentation;
use std::hash::DefaultHasher;
use std::hash::Hash;
use std::collections::VecDeque;
use dashmap::{DashSet};
use std::io::BufRead;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Error, Result};
use rayon::prelude::*;
use serde_json;
use serde_json::Value;
use clap::{Parser, Subcommand};


use mj_io::{
    build_pbar, expand_dirs, read_pathbuf_to_mem, write_mem_to_pathbuf,
};


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

        #[arg(required = true, long)]
        input_dir: PathBuf,

        #[arg(required = true, long)]
        output_dir: PathBuf,

        #[arg[required = true, long]]
        ngrams: usize,
    },

}

/*=========================================================
=                     DECONTAMINATION BLOCK               =
=========================================================*/

fn dencontam(reference_dir: &PathBuf, input_dir: &PathBuf, output_dir: &PathBuf, ngram_size: usize) -> Result<(), Error> {
    let start_main = Instant::now();
    println!("Starting decontam!");

    // First build the reference pool
    let start_ref = Instant::now();
    let reference_ngram_set: DashSet<usize> = DashSet::new();
    let ref_files = expand_dirs(vec![reference_dir.clone()], None).unwrap();
    let pbar = build_pbar(ref_files.len(), "Ref files");
    ref_files.into_par_iter().for_each(|p| {
        let contents = read_pathbuf_to_mem(&p).unwrap();
        for line in contents.lines() {
            let line = line.unwrap();
            let json_line: Value = serde_json::from_str(&line).unwrap();
            let text = json_line.get("text").unwrap().as_str().unwrap();
            let ngrams = get_ngrams(&text, ngram_size).unwrap();
            for ngram in ngrams {
                reference_ngram_set.insert(ngram);
            }
        }
        pbar.inc(1);
    });
    println!("Built reference lookup in {:?} secs", start_ref.elapsed().as_secs());


    // Then loop through and collect to see if anything matches
    println!("Starting decontamination");
    let total_docs = AtomicUsize::new(0);
    let contam_docs = AtomicUsize::new(0);
    let start_decontam = Instant::now();
    let bad_ids: DashSet<String> = DashSet::new();
    let input_paths = expand_dirs(vec![input_dir.clone()], None).unwrap();
    let pbar = build_pbar(input_paths.len(), "input paths");
    input_paths.into_par_iter().for_each(|p| {
        let mut path_docs = 0;
        let mut path_contam_docs = 0;
        let contents = read_pathbuf_to_mem(&p).unwrap();
        for line in contents.lines() {
            path_docs += 1;
            let line = line.unwrap();
            let json_line: Value = serde_json::from_str(&line).unwrap();
            let text = json_line.get("text").unwrap().as_str().unwrap();
            let id = json_line.get("id").unwrap().as_str().unwrap().to_string();
            let ngrams = get_ngrams(&text, ngram_size).unwrap();
            if ngrams.into_iter().any(|ngram| reference_ngram_set.contains(&ngram)) {
                path_contam_docs += 1;
                bad_ids.insert(id);
            }
        }
        total_docs.fetch_add(path_docs, Ordering::SeqCst);
        contam_docs.fetch_add(path_contam_docs, Ordering::SeqCst);
        pbar.inc(1);
    });

    println!("Finished document scan in {:?} seconds", start_decontam.elapsed().as_secs());
    let bad_ids: Vec<String> = bad_ids.into_iter().par_bridge().map(|p| p).collect();
    let bad_ids_json : Vec<u8> = serde_json::to_vec(&bad_ids).unwrap();
    let output_file = output_dir.clone().join("contaminated_ids.json");
    write_mem_to_pathbuf(&bad_ids_json, &output_file).unwrap();

    let total_docs = total_docs.into_inner();
    let contam_docs = contam_docs.into_inner();
    println!("Finished decontam in {:?} seconds", start_main.elapsed().as_secs());
    println!("Saw {:?} contam out of {:?} docs for a contam rate of {:.2}%", 
             contam_docs, total_docs, (contam_docs as f32 / total_docs as f32) * 100.0);
    Ok(())
}



fn get_ngrams(text_str: &str, ngram_size: usize) -> Result<Vec<usize>, Error> {

    let mut ngram: VecDeque<usize> = VecDeque::with_capacity(ngram_size);
    let mut full_ngrams: Vec<usize> = Vec::new();
    text_str.split_word_bounds().for_each(|w| {
        let word_hash = hash_object::<str>(w);
        ngram.push_back(word_hash);
        if ngram.len() >= ngram_size {
            let ngram_hash = hash_object::<VecDeque<usize>>(&ngram);
            full_ngrams.push(ngram_hash);
            ngram.pop_front();
        }
    });
    Ok(full_ngrams)
}

fn hash_object<T: Hash + ?Sized>(obj: &T) -> usize {
    let mut hasher = DefaultHasher::new();
    obj.hash(&mut hasher);
    hasher.finish() as usize    
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
            input_dir,
            output_dir,
            ngrams,        
        } => dencontam(reference_dir, input_dir, output_dir, *ngrams),
    };
    result.unwrap();
}


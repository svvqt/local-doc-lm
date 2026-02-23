//     for i in range(0, len(content), chunk_size-overlap):
//        chunks.append(content[i: i + chunk_size])
use pyo3::prelude::*;


#[pyfunction]
fn chunk_text(content: &str, chunk_size: usize, overlap: usize) -> PyResult<Vec<String>>{
    let mut chunks = Vec::new();
    let chars: Vec<char> = content.chars().collect(); 
    let n = chars.len();
    for i in (0..n).step_by(chunk_size-overlap){
        let chunk: String = chars[i..(i+chunk_size).min(n)].iter().collect();
        chunks.push(chunk);
    }
    Ok(chunks)
}


#[pymodule]
fn rag_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(chunk_text, m)?)?;
    Ok(())
}
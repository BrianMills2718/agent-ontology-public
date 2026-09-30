# MapReduce Document Processor

The MapReduce agent processes a large document by splitting it into chunks, analyzing each chunk independently in parallel, and then combining the results into a unified output. This architecture is directly inspired by the MapReduce programming model and is well-suited for tasks like summarization, entity extraction, and sentiment analysis over long documents.

The system accepts a MapReduceInput containing the full document text, the analysis task description (e.g., "Extract all named entities" or "Summarize key findings"), and a configurable chunk size in tokens.

Processing begins at the chunking step, which splits the input document into overlapping chunks. Each chunk has a chunk_id, the text content, and positional metadata (start and end offsets). The chunker ensures overlap between adjacent chunks to prevent information loss at boundaries. This step outputs a ChunkList.

Next, the system fans out to process all chunks in parallel. A map agent receives each chunk along with the analysis task description and produces a ChunkResult containing extracted information, a relevance score (0 to 1), and any chunk-specific metadata. This is a true parallel fan-out since chunks are independent.

After all chunks are processed, a shuffle step groups and deduplicates the chunk results. It identifies overlapping or duplicate findings across adjacent chunks (using the overlap regions) and merges them, producing a DeduplicatedResults list.

A reduce agent then receives all deduplicated results and synthesizes them into a single coherent output. It reconciles conflicting findings, establishes cross-chunk references, and produces a ReduceOutput with the synthesized analysis, a confidence score, and a provenance list linking each finding back to its source chunk.

Finally, a quality check gate evaluates the reduce output. If the confidence score is below 0.7, the system loops back to the reduce step with feedback asking for improvement. If confidence is adequate or after 2 retry attempts, the output is finalized. The final output is a MapReduceOutput containing the synthesized analysis, the provenance list, overall confidence, and processing statistics (number of chunks, total tokens processed, duration).

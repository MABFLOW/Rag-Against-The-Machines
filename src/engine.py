from .chunker import Chunker
from .indexer import Indexer
from .generator import Generator
from .parser import Parser
from .models import (
    AnsweredQuestion,
    MinimalAnswer,
    MinimalSource,
    StudentSearchResultsAndAnswer,
)
from pathlib import Path
from tqdm import tqdm
from .validation import Validation


class Engine:
    """Coordinates chunking, indexing, search, and answer generation."""

    def __init__(self, file: str = "data/processed/chunks.json") -> None:
        """Initializes the Engine.

        Args:
            file: Path to the JSON file of chunks used to build and query
                the search index.
        """
        self.file = file
        self.parser = Parser()
        self.validator = Validation()
        self.indexer = Indexer(self.file)

    def index(self, max_chunk_size: int = 2000) -> None:
        """Chunks the configured source folder and builds a search index.

        Args:
            max_chunk_size: Maximum number of characters allowed per chunk.
        """
        self.parser.validate_number(max_chunk_size, "max_chunk_size")
        if max_chunk_size > 2000:
            max_chunk_size = 2000
            print("Max_chunk_size has been reseted to 2000.")
        chunker = Chunker(max_chunk_size=max_chunk_size)
        chunker.run()
        self.parser.validate_file(self.file, "chunking file")

        self.indexer.index()

    def search(self, query: str, k: int) -> None:
        """Searches the index for a query and prints the matched sources.

        Args:
            query: The search query text.
            k: Number of top results to retrieve.
        """
        results = []
        self.parser.validate_number(k, "k")
        self.indexer.load()
        search_results = self.indexer.search(query, k)
        for source in search_results.search_results[0].retrieved_sources:
            path = source.file_path
            first_char = source.first_character_index
            last_char = source.last_character_index

            results.append(f"{path} [{first_char}:{last_char}]")
        return results

    def search_dataset(
        self, dataset_path: str, k: int, save_directory: str
    ) -> str:
        """Runs search over every question in a dataset and saves results.

        Args:
            dataset_path: Path to a JSON file containing a ``RagDataset``.
            k: Number of top results to retrieve per question.
            save_directory: Directory to write the search results to.
        """
        self.parser.validate_file(dataset_path, "dataset_path")
        save_dir = self.parser.validate_dir(save_directory, "save_directory")
        self.parser.validate_number(k, "k")

        self.indexer = Indexer(self.file)
        self.indexer.load()

        dataset = self.parser.load_data(dataset_path)

        rag_dataset = self.validator.validate_rag_dataset(dataset)

        queries = [item.question for item in rag_dataset.rag_questions]
        ids = [item.question_id for item in rag_dataset.rag_questions]

        search_results = self.indexer.search(queries, k, ids)

        output_filename = f"StudentSearchResults_{Path(dataset_path).stem}.json"
        self.parser.dump_to_dir(
            save_dir / output_filename, search_results)
        return (f"Saved student_search_results to \
                {save_directory}/{output_filename}")

    def answer(self, query: str, k: int) -> str:
        """Searches the index and generates an answer to a query.

        Args:
            query: The question to answer.
            k: Number of top results to retrieve as context.

        Returns:
            The generated answer text.
        """
        self.parser.validate_argument(query, 'query')
        self.parser.validate_number(k, 'k')
        self.generator = Generator()
        self.indexer.load()

        search_results = self.indexer.search(query, k)

        sources = search_results.search_results[0].retrieved_sources
        q_id = search_results.search_results[0].question_id

        contexts = []
        for source in sources:
            content = self.parser.read_from_file(source.file_path)
            first = source.first_character_index
            last = source.last_character_index
            contexts.append(
                content[first:last]
            )

        answer_text = self.generator.generate(query, contexts)
        minimal_answer = MinimalAnswer(
            question_id=q_id,
            question=query,
            retrieved_sources=sources,
            answer=answer_text
        )

        return {"answer": minimal_answer.answer, "sources": minimal_answer.retrieved_sources}

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ) -> None:
        """Generates answers for a dataset of prior search results.

        Args:
            student_search_results_path: Path to a JSON file containing a
                ``StudentSearchResults`` instance.
            save_directory: Directory to write the generated answers to.
        """
        self.generator = Generator()
        self.parser.validate_file(
            student_search_results_path,
            "student_search_results_path"
        )
        save_dir = self.parser.validate_dir(save_directory, "save_directory")

        results = self.validator.validate_student_search_results(
            self.parser.load_data(student_search_results_path)
        )

        answers = []

        for item in tqdm(results.search_results, desc="Answering dataset"):
            contexts = []

            for source in item.retrieved_sources:
                content = self.parser.read_from_file(source.file_path)
                first = source.first_character_index
                last = source.last_character_index

                contexts.append(content[first:last])

            answer = self.generator.generate(item.question, contexts)
            answers.append(MinimalAnswer(
                question_id=item.question_id,
                question=item.question,
                retrieved_sources=item.retrieved_sources,
                answer=answer
            ))

        final_output = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=results.k
        )

        self.parser.dump_to_dir(
            save_dir / "StudentSearchResultsAndAnswer.json", final_output)

    def evaluate(
        self, student_search_results_path: str, dataset_path: str
    ) -> float:
        """Computes recall@k of retrieved sources against ground truth.

        Args:
            student_search_results_path: Path to a JSON file containing a
                ``StudentSearchResults`` instance.
            dataset_path: Path to a JSON file containing a ``RagDataset``
                with ground-truth sources.

        Returns:
            The overall recall@k across all evaluated questions.
        """
        self.parser.validate_file(
            student_search_results_path, "student_search_results_path")
        self.parser.validate_file(dataset_path, "dataset_path")

        results = self.validator.validate_student_search_results(
            self.parser.load_data(student_search_results_path)
        )
        dataset = self.validator.validate_rag_dataset(
            self.parser.load_data(dataset_path)
        )

        ground_truth = {
            question.question_id: question
            for question in dataset.rag_questions
        }

        recalls = []
        for res in results.search_results:
            truth = ground_truth.get(res.question_id)
            if (
                truth is None
                or not isinstance(truth, AnsweredQuestion)
                or not truth.sources
            ):
                continue

            found = sum(
                any(self._sources_overlap(retrieved, source)
                    for retrieved in res.retrieved_sources)
                for source in truth.sources
                )
            recalls.append(found / len(truth.sources))

        overall_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"Recall@k over {len(recalls)} questions: {overall_recall:.4f}")
        return overall_recall

    def _sources_overlap(
        self, retrieved: MinimalSource, truth: MinimalSource
    ) -> bool:
        """Checks whether two source spans overlap in the same file.

        Args:
            retrieved: A retrieved source span.
            truth: A ground-truth source span.

        Returns:
            True if both spans reference the same file and their character
            ranges overlap, False otherwise.
        """
        if retrieved.file_path != truth.file_path:
            return False
        return (
            retrieved.first_character_index < truth.last_character_index
            and truth.first_character_index < retrieved.last_character_index
        )

from .chunker import Chunker
from .indexer import Indexer
from .generator import Generator
from .parser import Parser
from .models import *
from tqdm import tqdm
from .exceptions import *


class Engine:

    def __init__(self, file="data/processed/chunks.json"):
        self.file = file
        self.parser = Parser()
        self.indexer = Indexer(self.file)


    
    def index(self, max_chunk_size=2000):
        
        self.parser.validate_number(max_chunk_size, "max_chunk_size")

        chunker = Chunker(max_chunk_size=max_chunk_size)
        chunker.run()

        self.indexer.index()

    
    def search(self, query, k):
        self.parser.validate_number(k, "k")
        
        self.indexer.load()

        search_results = self.indexer.search(query, k)

        for source in search_results.search_results[0].retrieved_sources:
            print(f"{source.file_path} [{source.first_character_index}:{source.last_character_index}]")
    

    
    def search_dataset(self, dataset_path, k, save_directory):
        self.parser.validate_file(dataset_path, "dataset_path")
        save_dir = self.parser.validate_dir(save_directory, "save_directory")
        self.parser.validate_number(k, "k")

        self.indexer = Indexer(self.file)
        self.indexer.load()

        dataset = self.parser.load_data(dataset_path)
        
        data_unaswered = self.parser.load_unanswered_question(dataset)
        
        rag_dataset = RagDataset(rag_questions=data_unaswered)

        queries = [item.question for item in rag_dataset.rag_questions]
        ids = [item.question_id for item in rag_dataset.rag_questions]

        search_results = self.indexer.search(queries, k, ids)

        self.parser.dump_to_dir(save_dir / "StudentSearchResults.json", search_results)
        

    
    def answer(self, query, k):
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
            f = source.first_character_index
            l = source.last_character_index
            contexts.append(
                content[f:l]
            )
        
        answer_text = self.generator.generate(query, contexts)
        minimal_answer = MinimalAnswer(question_id=q_id, question_str=query, retrieved_sources=sources, answer=answer_text)

        return minimal_answer.answer


    def answer_dataset(self, student_search_results_path, save_directory):
        self.generator = Generator()
        self.parser.validate_file(student_search_results_path, "student_search_results_path")
        save_dir = self.parser.validate_dir(save_directory, "save_directory")

        output = []

        results = self.parser.load_data(student_search_results_path)

        try:
            StudentSearchResults(search_results=results["search_results"], k=results['k'])
        except Exception:
            raise ParsingError("Data in 'student_search_results_path' must be validated as the following:\nSearch_results: List[MinimalSearchResults]\nK: int")

        for res in results['search_results']:
            question_id = res['question_id']
            question = res['question_str']
            retrieved_sources = res['retrieved_sources']
            sources = []

            for src in retrieved_sources:

                file_path = src['file_path']
                first_char = src['first_character_index']
                last_char = src['last_character_index']
                sources.append(MinimalSource(file_path=file_path, first_character_index=first_char, last_character_index=last_char))

            output.append(MinimalSearchResults(
                question_id=question_id, 
                question_str=question, 
                retrieved_sources=sources
            ))
        answers = []
        for item in tqdm(output, desc="Answering dataset"):
            contexts = []
        
            for source in item.retrieved_sources:
                content = self.parser.read_from_file(source.file_path)
                f = source.first_character_index
                l = source.last_character_index
                
                contexts.append(content[f:l])

            answer = self.generator.generate(item.question_str, contexts)
            answers.append(MinimalAnswer(
                question_id=item.question_id,
                question_str=item.question_str,
                retrieved_sources=item.retrieved_sources,
                answer=answer
            ))

        final_output = StudentSearchResultsAndAnswer(search_results=answers, k=results['k'])

        self.parser.dump_to_dir(save_dir / "StudentSearchResultsAndAnswer.json" , final_output)


    def evaluate(self, student_search_results_path, dataset_path):
        self.parser.validate_file(student_search_results_path)
        self.parser.validate_file(dataset_path)

        results = self.parser.load_data(student_search_results_path)
        dataset = self.parser.load_data(dataset_path)
        
        ground_truth = {}
        for item in dataset['rag_questions']:
            rag_sources = []
            for src in item['sources']:
                rag_sources.append(
                    MinimalSource(
                        file_path=src['file_path'],
                        first_character_index=src['first_character_index'],
                        last_character_index=src['last_character_index']
                    )
                )
            ground_truth[item['question_id']] = AnsweredQuestion(
                question_id=item['question_id'],
                question=item['question'],
                sources=rag_sources,
                answer=item['answer']
            )

        recalls = []
        for res in results['search_results']:
            q_id = res['question_id']
            if q_id not in ground_truth:
                continue

            truth_srcs = ground_truth[q_id].sources
            retrieved_sources = res['retrieved_sources']

            if not truth_srcs:
                continue

            found = 0
            for truth in truth_srcs:
                if any(self._sources_overlap(r, truth) for r in retrieved_sources):
                    found += 1

            recalls.append(found / len(truth_srcs))

        overall_recall = sum(recalls) / len(recalls) if recalls else 0.0
        print(f"Recall@k over {len(recalls)} questions: {overall_recall:.4f}")
        return overall_recall

    def _sources_overlap(self, retrieved, truth):
        if retrieved['file_path'] != truth.file_path:
            return False
        return (
            retrieved['first_character_index'] < truth.last_character_index
            and truth.first_character_index < retrieved['last_character_index']
        )

        



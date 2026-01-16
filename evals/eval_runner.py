"""
Main Evaluation Runner
Runs all evaluation tests and generates comprehensive metrics
"""

import os
import json
import time
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
from rag_engine import ResumeRAG
from evals.eval_config import (
    INTENT_TEST_CASES, 
    RETRIEVAL_TEST_CASES, 
    E2E_TEST_CASES,
    GROUND_TRUTH_RESUMES,
    EVAL_METRICS_CONFIG
)

load_dotenv()

class RAGEvaluator:
    def __init__(self, rag_engine: ResumeRAG):
        self.rag = rag_engine
        self.results = {
            "timestamp": datetime.now().isoformat(),
            "intent_classification": {},
            "retrieval_quality": {},
            "response_quality": {},
            "extraction_accuracy": {},
            "summary": {}
        }
    
    def evaluate_intent_classification(self) -> Dict[str, Any]:
        """Test if the intent classifier correctly identifies query types"""
        print("\n" + "="*60)
        print("📋 EVALUATING INTENT CLASSIFICATION")
        print("="*60)
        
        correct = 0
        total = len(INTENT_TEST_CASES)
        detailed_results = []
        
        for i, test_case in enumerate(INTENT_TEST_CASES, 1):
            query = test_case["query"]
            expected_intent = test_case["expected_intent"]
            
            print(f"\n[{i}/{total}] Testing: '{query}'")
            
            try:
                intent_result = self.rag.classify_intent(query)
                predicted_intent = intent_result.get("type", "unknown")
                
                is_correct = predicted_intent == expected_intent
                if is_correct:
                    correct += 1
                    print(f"✅ PASS - Intent: {predicted_intent}")
                else:
                    print(f"❌ FAIL - Expected: {expected_intent}, Got: {predicted_intent}")
                
                detailed_results.append({
                    "query": query,
                    "expected": expected_intent,
                    "predicted": predicted_intent,
                    "correct": is_correct,
                    "full_intent": intent_result
                })
                
                time.sleep(1)  # Rate limit protection
                
            except Exception as e:
                print(f"⚠️  ERROR: {str(e)}")
                detailed_results.append({
                    "query": query,
                    "expected": expected_intent,
                    "predicted": "ERROR",
                    "correct": False,
                    "error": str(e)
                })
        
        accuracy = correct / total if total > 0 else 0
        threshold = EVAL_METRICS_CONFIG["intent"]["accuracy_threshold"]
        
        results = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "passed_threshold": accuracy >= threshold,
            "threshold": threshold,
            "detailed_results": detailed_results
        }
        
        print(f"\n{'='*60}")
        print(f"Intent Classification Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"Threshold: {threshold:.2%} - {'✅ PASSED' if results['passed_threshold'] else '❌ FAILED'}")
        print(f"{'='*60}")
        
        self.results["intent_classification"] = results
        return results
    
    def evaluate_retrieval_quality(self) -> Dict[str, Any]:
        """Test if the right resumes are being retrieved"""
        print("\n" + "="*60)
        print("🔍 EVALUATING RETRIEVAL QUALITY")
        print("="*60)
        
        detailed_results = []
        precision_scores = []
        recall_scores = []
        
        for i, test_case in enumerate(RETRIEVAL_TEST_CASES, 1):
            query = test_case["query"]
            expected_resumes = set(test_case["expected_resumes"])
            
            print(f"\n[{i}/{len(RETRIEVAL_TEST_CASES)}] Testing: '{query}'")
            
            try:
                # Get retrieval results based on intent
                intent = self.rag.classify_intent(query)
                intent_type = intent.get('type', 'general')
                
                if intent_type == 'list_filter':
                    _, docs = self.rag.handle_list_query(query, intent)
                elif intent_type == 'compare_specific':
                    _, docs = self.rag.handle_compare_query(query, intent)
                else:
                    _, docs = self.rag.handle_general_query(query)
                
                # Extract retrieved filenames
                retrieved_resumes = set()
                for doc in docs:
                    if isinstance(doc, str):
                        # Parse from summary string
                        continue  # Skip for list mode summaries
                    else:
                        source = doc.metadata.get('source', '')
                        if source:
                            retrieved_resumes.add(source)
                
                # Calculate metrics
                true_positives = len(expected_resumes & retrieved_resumes)
                false_positives = len(retrieved_resumes - expected_resumes)
                false_negatives = len(expected_resumes - retrieved_resumes)
                
                precision = true_positives / len(retrieved_resumes) if retrieved_resumes else 0
                recall = true_positives / len(expected_resumes) if expected_resumes else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                
                print(f"  Expected: {expected_resumes}")
                print(f"  Retrieved: {retrieved_resumes}")
                print(f"  Precision: {precision:.2%}, Recall: {recall:.2%}, F1: {f1:.2%}")
                
                detailed_results.append({
                    "query": query,
                    "expected_resumes": list(expected_resumes),
                    "retrieved_resumes": list(retrieved_resumes),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                })
                
                time.sleep(1)
                
            except Exception as e:
                print(f"⚠️  ERROR: {str(e)}")
                detailed_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
        
        precision_threshold = EVAL_METRICS_CONFIG["retrieval"]["precision_threshold"]
        recall_threshold = EVAL_METRICS_CONFIG["retrieval"]["recall_threshold"]
        
        results = {
            "average_precision": avg_precision,
            "average_recall": avg_recall,
            "average_f1": avg_f1,
            "passed_precision_threshold": avg_precision >= precision_threshold,
            "passed_recall_threshold": avg_recall >= recall_threshold,
            "precision_threshold": precision_threshold,
            "recall_threshold": recall_threshold,
            "detailed_results": detailed_results
        }
        
        print(f"\n{'='*60}")
        print(f"Average Precision: {avg_precision:.2%} (Threshold: {precision_threshold:.2%})")
        print(f"Average Recall: {avg_recall:.2%} (Threshold: {recall_threshold:.2%})")
        print(f"Average F1 Score: {avg_f1:.2%}")
        print(f"{'='*60}")
        
        self.results["retrieval_quality"] = results
        return results
    
    def evaluate_response_quality(self) -> Dict[str, Any]:
        """Test end-to-end response quality"""
        print("\n" + "="*60)
        print("💬 EVALUATING RESPONSE QUALITY")
        print("="*60)
        
        detailed_results = []
        quality_scores = []
        
        for i, test_case in enumerate(E2E_TEST_CASES, 1):
            query = test_case["query"]
            quality_checks = test_case["quality_checks"]
            
            print(f"\n[{i}/{len(E2E_TEST_CASES)}] Testing: '{query}'")
            
            try:
                response, intent, docs = self.rag.answer_query(query)
                
                # Check for required content
                must_contain = quality_checks.get("must_contain", [])
                must_not_contain = quality_checks.get("must_not_contain", [])
                
                response_lower = response.lower()
                
                contains_score = sum(1 for item in must_contain if item.lower() in response_lower) / len(must_contain) if must_contain else 1
                not_contains_score = sum(1 for item in must_not_contain if item.lower() not in response_lower) / len(must_not_contain) if must_not_contain else 1
                
                overall_score = (contains_score + not_contains_score) / 2
                quality_scores.append(overall_score)
                
                print(f"  Contains Required: {contains_score:.2%}")
                print(f"  Avoids Unwanted: {not_contains_score:.2%}")
                print(f"  Overall Quality: {overall_score:.2%}")
                
                detailed_results.append({
                    "query": query,
                    "response": response[:500],  # Truncate for storage
                    "contains_score": contains_score,
                    "not_contains_score": not_contains_score,
                    "overall_score": overall_score,
                    "intent": intent
                })
                
                time.sleep(2)  # Rate limit protection
                
            except Exception as e:
                print(f"⚠️  ERROR: {str(e)}")
                detailed_results.append({
                    "query": query,
                    "error": str(e)
                })
        
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        threshold = EVAL_METRICS_CONFIG["response_quality"]["relevance_threshold"]
        
        results = {
            "average_quality_score": avg_quality,
            "passed_threshold": avg_quality >= threshold,
            "threshold": threshold,
            "detailed_results": detailed_results
        }
        
        print(f"\n{'='*60}")
        print(f"Average Response Quality: {avg_quality:.2%} (Threshold: {threshold:.2%})")
        print(f"{'='*60}")
        
        self.results["response_quality"] = results
        return results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate overall evaluation summary"""
        intent_passed = self.results["intent_classification"].get("passed_threshold", False)
        retrieval_precision_passed = self.results["retrieval_quality"].get("passed_precision_threshold", False)
        retrieval_recall_passed = self.results["retrieval_quality"].get("passed_recall_threshold", False)
        response_passed = self.results["response_quality"].get("passed_threshold", False)
        
        total_tests = 4
        tests_passed = sum([intent_passed, retrieval_precision_passed, retrieval_recall_passed, response_passed])
        
        summary = {
            "overall_pass_rate": tests_passed / total_tests,
            "tests_passed": tests_passed,
            "total_tests": total_tests,
            "component_status": {
                "intent_classification": "✅ PASSED" if intent_passed else "❌ FAILED",
                "retrieval_precision": "✅ PASSED" if retrieval_precision_passed else "❌ FAILED",
                "retrieval_recall": "✅ PASSED" if retrieval_recall_passed else "❌ FAILED",
                "response_quality": "✅ PASSED" if response_passed else "❌ FAILED"
            }
        }
        
        self.results["summary"] = summary
        return summary
    
    def run_all_evaluations(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        print("\n" + "🚀 " + "="*58)
        print("🚀  STARTING COMPREHENSIVE RAG EVALUATION")
        print("🚀 " + "="*58)
        
        self.evaluate_intent_classification()
        self.evaluate_retrieval_quality()
        self.evaluate_response_quality()
        summary = self.generate_summary()
        
        print("\n" + "📊 " + "="*58)
        print("📊  EVALUATION SUMMARY")
        print("📊 " + "="*58)
        print(f"\nOverall Pass Rate: {summary['overall_pass_rate']:.2%} ({summary['tests_passed']}/{summary['total_tests']})")
        print("\nComponent Status:")
        for component, status in summary["component_status"].items():
            print(f"  {component}: {status}")
        print("\n" + "="*60)
        
        # Save results
        self.save_results()
        
        return self.results
    
    def save_results(self, output_dir="evals/eval_results"):  
        """Save evaluation results to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"eval_results_{timestamp}.json")
        
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filepath}")


def run_evaluation():
    """Main function to run evaluation"""
    if not os.path.exists("./db"):
        print("❌ Error: No database found. Please run ingestion first.")
        return
    
    print("Loading RAG engine...")
    rag_engine = ResumeRAG()
    
    evaluator = RAGEvaluator(rag_engine)
    results = evaluator.run_all_evaluations()
    
    return results


if __name__ == "__main__":
    run_evaluation()
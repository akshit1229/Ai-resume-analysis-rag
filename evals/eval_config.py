"""
Evaluation Configuration File
Define test cases, ground truth data, and expected behaviors for RAG evaluation
"""

# Sample Ground Truth Metadata (replace with your actual resume data)
GROUND_TRUTH_RESUMES = [
    {
        "filename": "Akshit__resume.pdf",
        "expected_metadata": {
            "name": "Akshit",
            "email": "akshit1229.doe@email.com",
            "years_exp": 0,
            "skills": ["python", "java"],
            "location": "Karnal",
            "education": ["B.Tech Computer Science from Chitkara University"]
        }
    },
    {
        "filename": "Resume-Dhanshree.pdf", 
        "expected_metadata": {
            "name": "Dhanshree",
            "email": "dhanubangale2003@gmail.com",
            "years_exp": 0,
            "skills": ["C", "JAVA", "python", "SQL","JAVA"],
            "location": "Pune",
            "education": ["Diploma - DR.D.Y.Patil Polytechnic nerul", "BE in CSE-IOT(CS&BC)"]
        }
    },
    {
        "filename": "RESUME-HiteshHinge.pdf",
        "expected_metadata": {
            "name": "HITESH HINGE",
            "email": "hitesh23.hinge@gmail.com",
            "years_exp": 8,
            "skills": ["Auditing", "Invoice Management", "Payments"],
            "location": "Pune",
            "education": ["B.Com. from Savitribai Phule Pune University, Nashik, Maharashtra in 2007"]
        }
    },
    {
        "filename": "RESUME-MANOJ_SHINDE-1.pdf",
        "expected_metadata": {
            "name": "Manoj Shinde",
            "email": "shindemanoj2886@gmail.com",
            "years_exp": 16,
            "skills": ["Account", "Finance"],
            "location": "Bangalore",
            "education": ["M.COM PUNE UNIVERCITY APR 2012/13 PASS CLASS","B.COM PUNE UNIVERCITY APR 2007/08 PASS CLASS"]
        }
    },
    {
        "filename": "Resume-NaveenBohra.pdf",
        "expected_metadata": {
            "name": "Manoj Shinde",
            "email": "naveen_bohra41@yahoo.com",
            "years_exp": 14,
            "skills": ["SAP", "Auditing","Accounts"],
            "location": "Pune",
            "education": ["M.COM PUNE UNIVERCITY APR 2012/13 PASS CLASS","B.COM PUNE UNIVERCITY APR 2007/08 PASS CLASS"]
        }
    },
   
]

# Intent Classification Test Cases
INTENT_TEST_CASES = [
    {
        "query": "List all Python developers in Bangalore",
        "expected_intent": "list_filter",
        "expected_keywords": ["python", "bangalore"]
    },
    {
        "query": "Find candidates with 5+ years of experience",
        "expected_intent": "list_filter",
        "expected_keywords": ["5 years", "experience"]
    },
    {
        "query": "Show me all B.Tech graduates",
        "expected_intent": "list_filter",
        "expected_keywords": ["b.tech"]
    },
    {
        "query": "Compare John Doe and Jane Smith",
        "expected_intent": "compare_specific",
        "expected_names": ["john doe", "jane smith"]
    },
    {
        "query": "Tell me about Harish's experience",
        "expected_intent": "compare_specific",
        "expected_names": ["harish"]
    },
    {
        "query": "What skills are most common among candidates?",
        "expected_intent": "general",
        "expected_keywords": []
    },
    {
        "query": "What is machine learning?",
        "expected_intent": "general",
        "expected_keywords": []
    }
]

# Retrieval Test Cases (what resumes should be retrieved for each query)
RETRIEVAL_TEST_CASES = [
    {
        "query": "Find Python developers",
        "expected_resumes": ["Resume-Dhanshree.pdf","Akshit__resume.pdf"], 
        "expected_min_count": 1
    },
    {
        "query": "List candidates in Pune",
        "expected_resumes": ["Resume-NaveenBohra.pdf"],
        "expected_min_count": 1
    },
    {
        "query": "Who has a Masters degree?",
        "expected_resumes": ["RESUME-MANOJ_SHINDE-1.pdf"],
        "expected_min_count": 1
    }, 
    
]

# End-to-End Response Quality Test Cases
E2E_TEST_CASES = [
    {
        "query": "List all Python developers in Bangalore",
        "quality_checks": {
            "must_contain": ["python", "bangalore"],
            "must_not_contain": ["pune", "java only"],
            "format_check": "list"  
        }
    },
    {
        "query": "Compare Akshit and Dhanshree  for a senior developer role",
        "quality_checks": {
            "must_contain": ["Akshit", "Dhanshree"],
            "must_not_contain": [],
            "format_check": "comparison" 
        }
    }
]

# Evaluation Metrics Configuration
EVAL_METRICS_CONFIG = {
    "extraction": {
        "fields_to_check": ["name", "email", "years_exp", "skills", "location", "education"],
        "required_accuracy": 0.8,  # 80% minimum accuracy
    },
    "retrieval": {
        "precision_threshold": 0.7,
        "recall_threshold": 0.6,
        "k_values": [3, 5, 10]  # Test retrieval at different k values
    },
    "intent": {
        "accuracy_threshold": 0.85
    },
    "response_quality": {
        "relevance_threshold": 0.7,
        "completeness_threshold": 0.6
    }
}

# Synthetic Test Data Generator Config
SYNTHETIC_QUERIES = {
    "list_filter_templates": [
        "Find all {skill} developers",
        "Show me candidates with {years}+ years of experience",
        "List people in {location}",
        "Who has {education} degree?",
        "Find {skill} experts in {location}"
    ],
    "compare_templates": [
        "Compare {name1} and {name2}",
        "How does {name1} compare to {name2}?",
        "Tell me about {name1} vs {name2}",
        "Which is better: {name1} or {name2}?"
    ],
    "general_templates": [
        "What skills are most common?",
        "How many candidates have {skill}?",
        "What's the average experience level?",
        "Tell me about the education backgrounds"
    ]
}
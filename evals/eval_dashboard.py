"""
Evaluation Dashboard
Streamlit app to visualize evaluation results
"""

import streamlit as st
import json
import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

st.set_page_config(page_title="RAG Evaluation Dashboard", layout="wide")

def load_latest_results(results_dir="./eval_results"):
    """Load the most recent evaluation results"""
    if not os.path.exists(results_dir):
        return None
    
    files = [f for f in os.listdir(results_dir) if f.startswith("eval_results_")]
    if not files:
        return None
    
    latest_file = sorted(files)[-1]
    filepath = os.path.join(results_dir, latest_file)
    
    with open(filepath, 'r') as f:
        return json.load(f)

def load_all_results(results_dir="./eval_results"):
    """Load all evaluation results for historical comparison"""
    if not os.path.exists(results_dir):
        return []
    
    files = [f for f in os.listdir(results_dir) if f.startswith("eval_results_")]
    results = []
    
    for file in sorted(files):
        filepath = os.path.join(results_dir, file)
        with open(filepath, 'r') as f:
            data = json.load(f)
            results.append(data)
    
    return results

def create_gauge_chart(value, title, threshold):
    """Create a gauge chart for metrics"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = value * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        delta = {'reference': threshold * 100, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'ticksuffix': "%"},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, threshold * 100], 'color': "lightgray"},
                {'range': [threshold * 100, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold * 100
            }
        }
    ))
    
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def display_intent_classification(results):
    """Display intent classification results"""
    st.header("📋 Intent Classification Results")
    
    intent_data = results["intent_classification"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Accuracy", f"{intent_data['accuracy']:.2%}", 
                  f"{intent_data['correct']}/{intent_data['total']}")
    
    with col2:
        st.metric("Threshold", f"{intent_data['threshold']:.2%}")
    
    with col3:
        status = "✅ PASSED" if intent_data['passed_threshold'] else "❌ FAILED"
        st.metric("Status", status)
    
    # Gauge chart
    fig = create_gauge_chart(intent_data['accuracy'], "Intent Classification Accuracy", 
                            intent_data['threshold'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results table
    with st.expander("📊 View Detailed Results"):
        df = pd.DataFrame(intent_data['detailed_results'])
        st.dataframe(df, use_container_width=True)
        
        # Confusion breakdown
        if 'predicted' in df.columns:
            intent_types = df['expected'].unique()
            confusion_data = []
            
            for expected in intent_types:
                for predicted in intent_types:
                    count = len(df[(df['expected'] == expected) & (df['predicted'] == predicted)])
                    confusion_data.append({
                        'Expected': expected,
                        'Predicted': predicted,
                        'Count': count
                    })
            
            confusion_df = pd.DataFrame(confusion_data)
            confusion_pivot = confusion_df.pivot(index='Expected', columns='Predicted', values='Count').fillna(0)
            
            fig = px.imshow(confusion_pivot, 
                           labels=dict(x="Predicted", y="Expected", color="Count"),
                           title="Intent Classification Confusion Matrix",
                           color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

def display_retrieval_quality(results):
    """Display retrieval quality results"""
    st.header("🔍 Retrieval Quality Results")
    
    retrieval_data = results["retrieval_quality"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Precision", f"{retrieval_data['average_precision']:.2%}")
    
    with col2:
        st.metric("Recall", f"{retrieval_data['average_recall']:.2%}")
    
    with col3:
        st.metric("F1 Score", f"{retrieval_data['average_f1']:.2%}")
    
    # Gauge charts
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_gauge_chart(retrieval_data['average_precision'], 
                                "Precision", 
                                retrieval_data['precision_threshold'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_gauge_chart(retrieval_data['average_recall'], 
                                "Recall", 
                                retrieval_data['recall_threshold'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    with st.expander("📊 View Detailed Results"):
        detailed_results = [r for r in retrieval_data['detailed_results'] if 'error' not in r]
        if detailed_results:
            df = pd.DataFrame(detailed_results)
            st.dataframe(df[['query', 'precision', 'recall', 'f1']], use_container_width=True)
            
            # Per-query performance chart
            fig = go.Figure()
            fig.add_trace(go.Bar(name='Precision', x=df['query'], y=df['precision']))
            fig.add_trace(go.Bar(name='Recall', x=df['query'], y=df['recall']))
            fig.add_trace(go.Bar(name='F1', x=df['query'], y=df['f1']))
            
            fig.update_layout(
                title="Per-Query Retrieval Metrics",
                xaxis_title="Query",
                yaxis_title="Score",
                barmode='group',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

def display_response_quality(results):
    """Display response quality results"""
    st.header("💬 Response Quality Results")
    
    response_data = results["response_quality"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Quality Score", f"{response_data['average_quality_score']:.2%}")
    
    with col2:
        st.metric("Threshold", f"{response_data['threshold']:.2%}")
    
    with col3:
        status = "✅ PASSED" if response_data['passed_threshold'] else "❌ FAILED"
        st.metric("Status", status)
    
    # Gauge chart
    fig = create_gauge_chart(response_data['average_quality_score'], 
                            "Response Quality Score", 
                            response_data['threshold'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed results
    with st.expander("📊 View Detailed Results"):
        detailed_results = [r for r in response_data['detailed_results'] if 'error' not in r]
        if detailed_results:
            for i, result in enumerate(detailed_results, 1):
                st.subheader(f"Query {i}: {result['query']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Contains Required", f"{result['contains_score']:.2%}")
                with col2:
                    st.metric("Avoids Unwanted", f"{result['not_contains_score']:.2%}")
                with col3:
                    st.metric("Overall", f"{result['overall_score']:.2%}")
                
                st.text_area("Response Preview", result['response'], height=100, key=f"response_{i}")
                st.divider()

def display_historical_trends(all_results):
    """Display historical performance trends"""
    st.header("📈 Historical Performance Trends")
    
    if len(all_results) < 2:
        st.info("Run evaluations multiple times to see historical trends")
        return
    
    # Extract metrics over time
    timestamps = []
    intent_accuracy = []
    precision = []
    recall = []
    response_quality = []
    
    for result in all_results:
        timestamps.append(result['timestamp'])
        intent_accuracy.append(result['intent_classification']['accuracy'])
        precision.append(result['retrieval_quality']['average_precision'])
        recall.append(result['retrieval_quality']['average_recall'])
        response_quality.append(result['response_quality']['average_quality_score'])
    
    # Create line chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=timestamps, y=intent_accuracy, 
                            mode='lines+markers', name='Intent Accuracy'))
    fig.add_trace(go.Scatter(x=timestamps, y=precision, 
                            mode='lines+markers', name='Precision'))
    fig.add_trace(go.Scatter(x=timestamps, y=recall, 
                            mode='lines+markers', name='Recall'))
    fig.add_trace(go.Scatter(x=timestamps, y=response_quality, 
                            mode='lines+markers', name='Response Quality'))
    
    fig.update_layout(
        title="RAG Performance Over Time",
        xaxis_title="Evaluation Run",
        yaxis_title="Score",
        yaxis=dict(tickformat=".0%"),
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)

def main():
    st.title("📊 RAG Evaluation Dashboard")
    st.markdown("---")
    
    # Load results
    latest_results = load_latest_results()
    
    if latest_results is None:
        st.error("No evaluation results found. Please run eval_runner.py first!")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("Evaluation Info")
        st.write(f"**Timestamp:** {latest_results['timestamp']}")
        
        st.markdown("---")
        
        summary = latest_results['summary']
        st.metric("Overall Pass Rate", 
                 f"{summary['overall_pass_rate']:.2%}",
                 f"{summary['tests_passed']}/{summary['total_tests']}")
        
        st.markdown("---")
        st.subheader("Component Status")
        for component, status in summary['component_status'].items():
            st.write(f"{status} {component.replace('_', ' ').title()}")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "Intent Classification", 
        "Retrieval Quality", 
        "Response Quality",
        "Historical Trends"
    ])
    
    with tab1:
        display_intent_classification(latest_results)
    
    with tab2:
        display_retrieval_quality(latest_results)
    
    with tab3:
        display_response_quality(latest_results)
    
    with tab4:
        all_results = load_all_results()
        display_historical_trends(all_results)

if __name__ == "__main__":
    main()
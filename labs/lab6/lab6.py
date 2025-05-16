import json
import re
import numpy as np
import pandas as pd
from datetime import datetime # For temporal analysis

# Scikit-learn imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# NLTK imports
from nltk.corpus import stopwords

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates # For formatting dates on plots

# --- Configuration ---
DATA_PATH = 'labs\lab6\construction_documents.json'
RANDOM_STATE = 1

# -------------------------------------------
# Data Loading and Initial Preprocessing
# -------------------------------------------

def load_data(json_path: str) -> pd.DataFrame:
    """Loads data from a JSON file into a Pandas DataFrame."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        # Attempt to convert 'date' column to datetime early on
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return df
    except FileNotFoundError:
        print(f"Error: The file {json_path} was not found.")
        return None

def preprocess_text_series(text_series: pd.Series) -> pd.Series:
    """Applies text preprocessing to a Pandas Series."""
    # Ensure stopwords are downloaded
    try:
        stop_words_list = stopwords.words('english')
    except LookupError:
        import nltk
        print("NLTK stopwords not found. Downloading...")
        nltk.download('stopwords')
        stop_words_list = stopwords.words('english')

    def clean_text(text: str) -> str:
        if not isinstance(text, str): return ""
        abbreviations = ['RFI', 'CO', 'SI', 'MEP', 'QA', 'QC', 'HVAC', 'PM', 'PO']
        for abbr in abbreviations:
            text = re.sub(rf'\b{abbr}\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'[\$€£]\d+(\.\d+)?', ' currencyamount ', text)
        text = re.sub(r'\d+\s*(days?|hours?|weeks?|months?|yrs?|min|sec)\b', ' timeunit ', text, flags=re.IGNORECASE)
        text = re.sub(r'\b[A-Z]{2,}-\d+\b', ' itemcode ', text)
        text = re.sub(r'\b\d+(\.\d+)?\s*(mm|cm|m|km|ft|inch|sqft|m2|m3|yd|lbs|kg)\b', ' measurementunit ', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+', ' numbertoken ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = text.lower()
        text = ' '.join([w for w in text.split() if w not in stop_words_list and len(w) > 1])
        return text.strip()
    return text_series.apply(clean_text)

# -------------------------------------------
# Visualization and Analysis Functions
# -------------------------------------------

def plot_doc_types_by_phase(df: pd.DataFrame):
    if 'project_phase' not in df.columns or 'document_type' not in df.columns: return
    df_vis = df.copy()
    df_vis['project_phase'] = df_vis['project_phase'].fillna('Unknown Phase')
    plt.figure(figsize=(12, 7))
    sns.countplot(data=df_vis, y='project_phase', hue='document_type', palette='viridis')
    plt.title('Distribution of Document Types Across Project Phases')
    plt.xlabel('Count'); plt.ylabel('Project Phase')
    plt.legend(title='Document Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(); plt.show()

def plot_confusion_matrix_func(y_true, y_pred, labels, title='Confusion Matrix'):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.title(title)
    plt.show()

def visualize_top_terms(pipeline: Pipeline, class_labels: list, num_terms: int = 10):
    try:
        column_transformer = pipeline.named_steps['preprocessor']
        text_transformer_step_name = None
        for name, transformer, _ in column_transformer.transformers_:
            if isinstance(transformer, (TfidfVectorizer, CountVectorizer)):
                text_transformer_step_name = name
                break
            elif isinstance(transformer, Pipeline) and isinstance(transformer.named_steps.get(list(transformer.named_steps.keys())[0]), (TfidfVectorizer, CountVectorizer)): # Check if it's a pipeline containing the vectorizer
                text_transformer_step_name = name # Name of the pipeline step in ColumnTransformer
                break
        
        if not text_transformer_step_name:
            print("Could not find TfidfVectorizer or CountVectorizer in the preprocessor.")
            return

        # Access the fitted vectorizer
        if isinstance(column_transformer.named_transformers_[text_transformer_step_name], Pipeline):
             # If the transformer itself is a pipeline (e.g., text processing steps then vectorizer)
            text_vectorizer_pipeline = column_transformer.named_transformers_[text_transformer_step_name]
            # Find the vectorizer within this sub-pipeline
            vectorizer_in_pipe_name = [name for name, step in text_vectorizer_pipeline.steps if isinstance(step, (TfidfVectorizer, CountVectorizer))][0]
            text_vectorizer = text_vectorizer_pipeline.named_steps[vectorizer_in_pipe_name]
        else:
            text_vectorizer = column_transformer.named_transformers_[text_transformer_step_name]

        classifier = pipeline.named_steps['classifier']
    except KeyError as e:
        print(f"KeyError accessing pipeline components: {e}. Ensure steps are named 'preprocessor' and 'classifier', and text vectorizer is identifiable.")
        return

    if not hasattr(classifier, 'feature_log_prob_') and not hasattr(classifier, 'coef_'):
        print(f"Classifier {type(classifier).__name__} unsupported for top terms visualization.")
        return

    feature_names = text_vectorizer.get_feature_names_out()
    num_text_features = len(feature_names)

    print("\nTop terms per document type:")
    for i, label in enumerate(class_labels):
        if hasattr(classifier, 'feature_log_prob_'): # Naive Bayes
            log_probs = classifier.feature_log_prob_[i, :num_text_features]
        elif hasattr(classifier, 'coef_'): # Logistic Regression, SVM with linear kernel
            if classifier.coef_.ndim > 1:
                log_probs = classifier.coef_[i, :num_text_features]
            else: # Binary case, or if coef_ is 1D for some reason
                log_probs = classifier.coef_[:num_text_features]
        elif hasattr(classifier, 'feature_importances_'): # Random Forest
            log_probs = classifier.feature_importances_[:num_text_features]
        else:
            continue
        
        top_n_indices = np.argsort(log_probs)[-num_terms:]
        top_n_terms = feature_names[top_n_indices]
        print(f"  {label}: {', '.join(top_n_terms[::-1])}")


def analyze_temporal_patterns(df: pd.DataFrame):
    """Analyzes and plots document type frequencies over time."""
    print("\n--- Temporal Pattern Analysis ---")
    if 'date' not in df.columns or 'document_type' not in df.columns:
        print("Missing 'date' or 'document_type' columns for temporal analysis.")
        return
    
    df_temp = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_temp['date']):
        df_temp['date'] = pd.to_datetime(df_temp['date'], errors='coerce')
    
    df_temp = df_temp.dropna(subset=['date', 'document_type'])
    if df_temp.empty:
        print("No valid date or document_type data for temporal analysis after cleaning.")
        return

    df_temp = df_temp.set_index('date')
    # Resample by month and count document types.
    try:
        monthly_counts = df_temp.groupby([pd.Grouper(freq='M'), 'document_type']).size().unstack(fill_value=0)
    except Exception as e:
        print(f"Error during resampling for temporal analysis: {e}")
        # Fallback if Grouper causes issues with specific pandas versions/data
        try:
            df_temp['month_year'] = df_temp.index.to_period('M')
            monthly_counts = df_temp.groupby(['month_year', 'document_type']).size().unstack(fill_value=0)
            monthly_counts.index = monthly_counts.index.to_timestamp() # Convert PeriodIndex back to DateTimeIndex for plotting
        except Exception as e_fallback:
            print(f"Fallback resampling also failed: {e_fallback}")
            return


    if monthly_counts.empty:
        print("No data to plot for temporal patterns after resampling.")
        return

    plt.figure(figsize=(14, 8))
    for col in monthly_counts.columns:
        plt.plot(monthly_counts.index, monthly_counts[col], marker='o', linestyle='-', label=col)
    
    plt.title('Document Type Frequency Over Time (Monthly)')
    plt.xlabel('Month')
    plt.ylabel('Number of Documents')
    plt.legend(title='Document Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m')) # Format x-axis dates
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(monthly_counts.index) // 12))) # Adjust tick frequency
    plt.tight_layout()
    plt.show()
    print("--- End of Temporal Pattern Analysis ---")

# -------------------------------------------
# Experiment Functions (Simplified for brevity, focus on main pipeline)
# -------------------------------------------
def experiment_feature_combinations(df: pd.DataFrame, target_names: list, default_imputation_strategy='most_frequent'):
    """Compares models with different feature combinations (text-only, meta-only, combined)."""
    print("\n--- Experiment: Feature Combinations ---")
    
    metadata_cols = ['project_phase', 'author_role']
    text_col = 'clean_content'
    target_col = 'document_type'

    if 'clean_content' not in df.columns:
        df['clean_content'] = preprocess_text_series(df['content'])

    X = df 
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)

    text_transformer_obj = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
    
    metadata_transformer_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=default_imputation_strategy)), 
        ('dict_converter', FunctionTransformer(lambda x: pd.DataFrame(x, columns=metadata_cols).to_dict(orient='records'))), 
        ('vectorizer', DictVectorizer(sparse=False))
    ])
    
    feature_sets_configs = {
        "Text Only": ColumnTransformer(
            [('text_features', text_transformer_obj, text_col)], remainder='drop'),
        "Metadata Only": ColumnTransformer(
            [('meta_features', metadata_transformer_pipeline, metadata_cols)], remainder='drop'),
        "Text + Metadata": ColumnTransformer(
            [
                ('text_features', text_transformer_obj, text_col),
                ('meta_features', metadata_transformer_pipeline, metadata_cols)
            ], remainder='drop')
    }

    for name, preprocessor_config in feature_sets_configs.items():
        print(f"\nTraining with features: {name}")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor_config),
            ('classifier', MultinomialNB())
        ])
        try:
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"  Accuracy: {accuracy:.4f}")
        except Exception as e:
            print(f"  Error during '{name}' experiment: {e}")
    print("--- End of Feature Combination Experiment ---")


def perform_cross_validation(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv_folds: int = 5):
    print(f"\n--- Cross-Validation ({cv_folds}-folds) ---")
    try:
        cv_scores = cross_val_score(pipeline, X, y, cv=cv_folds, scoring='accuracy')
        print(f"  CV Accuracy Scores: {cv_scores}")
        print(f"  Mean CV Accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores) * 2:.4f})")
    except Exception as e:
        print(f"  Error during cross-validation: {e}")
    print("--- End of Cross-Validation ---")

# -------------------------------------------
# Simple Interface for New Document Classification
# -------------------------------------------
def classify_new_document(pipeline: Pipeline, document_text: str, metadata: dict) -> str:
    """Classifies a new document using the trained pipeline."""
    # Preprocess the text
    cleaned_text = preprocess_text_series(pd.Series([document_text])).iloc[0]
    
    data_dict = {'clean_content': [cleaned_text]}
    for key, val in metadata.items():
        data_dict[key] = [val if val else None] # Use None for missing metadata

    metadata_cols_in_pipeline = ['project_phase', 'author_role'] # As defined in main preprocessor
    for col in metadata_cols_in_pipeline:
        if col not in data_dict:
            data_dict[col] = [None]
            
    new_doc_df = pd.DataFrame(data_dict)
    
    try:
        prediction_probabilities = pipeline.predict_proba(new_doc_df)
        predicted_class_index = np.argmax(prediction_probabilities, axis=1)[0]
        predicted_class_label = pipeline.classes_[predicted_class_index]
        confidence_score = prediction_probabilities[0, predicted_class_index]
        
        print(f"    Confidence: {confidence_score:.4f}")
        return predicted_class_label
    except Exception as e:
        print(f"Error during prediction for new document: {e}")
        print("Ensure metadata keys match those used in training ('project_phase', 'author_role').")
        import traceback
        traceback.print_exc()
        return "Error in prediction"

def experiment_imputation_strategies(df: pd.DataFrame, target_names: list):
    """Compares different imputation strategies for metadata."""
    print("\n--- Experiment: Imputation Strategies for Metadata ---")
    metadata_cols = ['project_phase', 'author_role']
    text_col = 'clean_content'
    target_col = 'document_type'

    if 'clean_content' not in df.columns:
        print("  'clean_content' column not found. Performing text preprocessing for this experiment.")
        df_exp = df.copy()
        df_exp['clean_content'] = preprocess_text_series(df_exp['content'])
    else:
        df_exp = df.copy()

    X_exp = df_exp # Pipeline will select columns
    y_exp = df_exp[target_col]
    
    # Split data for this experiment
    X_train_exp, X_test_exp, y_train_exp, y_test_exp = train_test_split(
        X_exp, y_exp, test_size=0.3, random_state=RANDOM_STATE, stratify=y_exp
    )

    # Strategies to test for categorical features
    strategies = ['most_frequent', 'constant'] 
    
    for strategy in strategies:
        print(f"\n  Testing imputation strategy: '{strategy}'")
        
        # Metadata processing pipeline
        current_metadata_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=strategy, 
                                     fill_value='Unknown' if strategy == 'constant' else None)),
            # Convert imputed numpy array back to DataFrame, then to list of dicts
            ('to_dicts', FunctionTransformer(
                lambda x: pd.DataFrame(x, columns=metadata_cols).to_dict(orient='records'))),
            ('vectorizer', DictVectorizer(sparse=False)) 
        ])

        # Preprocessor combining text and metadata handling for this experiment
        exp_preprocessor = ColumnTransformer(
            transformers=[
                ('text_vec', TfidfVectorizer(max_features=500, ngram_range=(1,1)), text_col), # Simpler TF-IDF for speed
                ('meta_proc', current_metadata_transformer, metadata_cols)
            ],
            remainder='drop'
        )

        exp_pipeline = Pipeline(steps=[
            ('preprocessor', exp_preprocessor),
            ('classifier', MultinomialNB(alpha=0.5)) # Basic classifier
        ])

        try:
            exp_pipeline.fit(X_train_exp, y_train_exp)
            y_pred_exp = exp_pipeline.predict(X_test_exp)
            accuracy = accuracy_score(y_test_exp, y_pred_exp)
            print(f"    Accuracy with '{strategy}' imputation: {accuracy:.4f}")
        except Exception as e:
            print(f"    Error with strategy '{strategy}': {e}")
            import traceback
            traceback.print_exc()
            
    print("--- End of Imputation Strategy Experiment ---")


# -------------------------------------------
# Main Pipeline Execution
# -------------------------------------------

def main():
    print("Starting Lab 6 Document Classification Pipeline...")
    df = load_data(DATA_PATH)
    if df is None: return

    if 'document_type' not in df.columns:
        print("Error: 'document_type' column not found."); return
    original_target_names = sorted(list(df['document_type'].astype(str).unique()))
        
    df['clean_content'] = preprocess_text_series(df['content'])

    plot_doc_types_by_phase(df.copy())
    
    analyze_temporal_patterns(df.copy()) # Temporal Analysis

    experiment_imputation_strategies(df.copy(), target_names=original_target_names) # Practice imputation
    
    experiment_feature_combinations(df.copy(), target_names=original_target_names) # Compare feature sets

    print("\n--- Main Model Training and Evaluation ---")
    metadata_cols = ['project_phase', 'author_role']
    text_col = 'clean_content'
    target_col = 'document_type'
    X = df
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y)
    
    main_metadata_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('dict_converter', FunctionTransformer(lambda x: pd.DataFrame(x, columns=metadata_cols).to_dict(orient='records'))),
        ('vectorizer', DictVectorizer(sparse=False))
    ])

    # Define text transformer separately to ensure consistent naming for visualize_top_terms
    text_transformer_main = TfidfVectorizer(max_features=1000, ngram_range=(1,2), min_df=2)

    main_preprocessor = ColumnTransformer(
        transformers=[
            ('text_vectorizer', text_transformer_main, text_col), # Explicit name
            ('metadata_processor', main_metadata_transformer, metadata_cols)
        ], remainder='drop' 
    )

    # --- Classifier Comparison ---
    classifiers_to_compare = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=RANDOM_STATE, max_iter=300),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, class_weight='balanced') # Added RF
    }

    trained_pipelines = {} # To store trained main pipeline for new doc classification

    for clf_name, classifier_obj in classifiers_to_compare.items():
        print(f"\n--- Training and Evaluating: {clf_name} ---")
        pipeline = Pipeline(steps=[
            ('preprocessor', main_preprocessor),
            ('classifier', classifier_obj)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        print(f"\n{clf_name} - Evaluation:")
        print(classification_report(y_test, y_pred, target_names=original_target_names, zero_division=0))
        plot_confusion_matrix_func(y_test, y_pred, labels=original_target_names, title=f'Confusion Matrix - {clf_name}')
        if clf_name != "Random Forest" or hasattr(classifier_obj, 'feature_importances_'): # RF needs feature_importances_ for this viz
             visualize_top_terms(pipeline, class_labels=original_target_names, num_terms=10)
        
        perform_cross_validation(pipeline, X.copy(), y.copy(), cv_folds=3) # Reduced folds for RF speed

        if clf_name == "Multinomial Naive Bayes": # Save one pipeline for new doc classification
            trained_pipelines['main_model'] = pipeline


    # Analyze Misclassifications
    if 'main_model' in trained_pipelines:
        nb_pipeline_for_misclass = trained_pipelines['main_model']
        y_pred_for_misclass = nb_pipeline_for_misclass.predict(X_test) # Re-predict if needed or use stored y_pred
        print("\nAnalyzing Misclassifications (Naive Bayes):")
        misclassified_indices = X_test[y_test != y_pred_for_misclass].index
        if not misclassified_indices.empty:
            print(f"Found {len(misclassified_indices)} misclassifications. Showing up to 3 examples:")
            for i, idx in enumerate(misclassified_indices[:3]):
                original_doc_info = df.loc[idx]
                print(f"  Example {i+1}: Doc ID: {original_doc_info.get('document_id', 'N/A')}, True: {y_test.loc[idx]}, Predicted: {y_pred_for_misclass[y_test.index.get_loc(idx)]}")
                print(f"    Content: {original_doc_info['content'][:100]}...")
        else:
            print("No misclassifications found with Naive Bayes on the test set!")


    # --- Simple Interface for Classifying New Documents ---
    if 'main_model' in trained_pipelines:
        main_pipeline = trained_pipelines['main_model'] # Use the trained Naive Bayes pipeline
        print("\n--- Classify New Document ---")
        while True:
            new_text = input("Enter document text (or type 'quit' to exit): ")
            if new_text.lower() == 'quit':
                break
            if not new_text.strip():
                print("Document text cannot be empty.")
                continue

            # Get metadata
            print("Enter metadata (press Enter to skip/use default):")
            proj_phase = input("  Project Phase (e.g., Design, Construction, Closeout): ").strip()
            auth_role = input("  Author Role (e.g., Project Manager, Site Engineer): ").strip()
            
            new_metadata = {
                'project_phase': proj_phase if proj_phase else None, # Use None if empty
                'author_role': auth_role if auth_role else None
            }
            
            predicted_type = classify_new_document(main_pipeline, new_text, new_metadata)
            print(f"==> Predicted Document Type: {predicted_type}\n")

    print("\nLab 6 script execution finished.")

if __name__ == "__main__":
    main()

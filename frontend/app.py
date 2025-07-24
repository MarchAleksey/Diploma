import streamlit as st
from utils.api_client import APIClient

# Configure the page
st.set_page_config(
    page_title="ML Noise Impact Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize API client
api_client = APIClient()


def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Dataset",
            "Model Selection",
            "Noise Configuration",
            "Training",
            "Results",
        ],
    )

    # Display the selected page
    if page == "Home":
        show_home_page()
    elif page == "Dataset":
        show_dataset_page()
    elif page == "Model Selection":
        show_model_page()
    elif page == "Noise Configuration":
        show_noise_page()
    elif page == "Training":
        show_training_page()
    elif page == "Results":
        show_results_page()


def show_home_page():
    st.title("ML Noise Impact Analysis")

    st.markdown(
        """
    ## Welcome to the ML Noise Impact Analysis Tool
    
    This application allows you to analyze the impact of noise in annotations on machine learning model training quality.
    
    ### Features:
    - Upload or select datasets
    - Choose from various machine learning models
    - Configure different types and levels of noise
    - Train models with the specified configurations
    - Visualize and analyze training results
    
    ### How to use:
    1. Start by selecting or uploading a dataset
    2. Choose a machine learning model
    3. Configure the type and level of noise
    4. Train the model
    5. Analyze the results
    
    ### About:
    This application was developed as part of a thesis project on "The Impact of Noise in Annotations on Machine Learning Model Training Quality."
    """
    )

    # Display some statistics if available
    try:
        stats = api_client.get_statistics()
        st.subheader("System Statistics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Available Datasets", stats["dataset_count"])
        with col2:
            st.metric("Available Models", stats["model_count"])
        with col3:
            st.metric("Completed Experiments", stats["experiment_count"])
    except Exception as e:
        st.warning(
            "Could not fetch system statistics. Make sure the backend is running."
        )


def show_dataset_page():
    st.title("Dataset Selection")

    # Dataset selection options
    st.subheader("Select a Dataset")

    dataset_option = st.radio(
        "Choose an option:", ["Use built-in dataset", "Upload your own dataset"]
    )

    if dataset_option == "Use built-in dataset":
        try:
            datasets = api_client.get_available_datasets()
            selected_dataset = st.selectbox("Select a dataset", datasets)

            if selected_dataset:
                dataset_info = api_client.get_dataset_info(selected_dataset)

                st.subheader("Dataset Information")
                st.write(f"Name: {dataset_info['name']}")
                st.write(f"Description: {dataset_info['description']}")
                st.write(f"Number of samples: {dataset_info['n_samples']}")
                st.write(f"Number of features: {dataset_info['n_features']}")
                st.write(f"Target type: {dataset_info['target_type']}")

                # Preview the dataset
                st.subheader("Dataset Preview")
                st.dataframe(dataset_info["preview"])

                # Save the selected dataset to session state
                if st.button("Use this dataset"):
                    st.session_state.dataset = selected_dataset
                    st.success(f"Dataset '{selected_dataset}' selected!")

        except Exception as e:
            st.error(f"Error loading datasets: {str(e)}")

    else:  # Upload dataset
        st.subheader("Upload a Dataset")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file is not None:
            try:
                # Preview the uploaded dataset
                import pandas as pd

                df = pd.read_csv(uploaded_file)
                st.subheader("Dataset Preview")
                st.dataframe(df.head())

                # Configure target column
                target_column = st.selectbox(
                    "Select target column", df.columns.tolist()
                )

                # Upload the dataset to the backend
                if st.button("Use this dataset"):
                    response = api_client.upload_dataset(uploaded_file, target_column)
                    if response["success"]:
                        st.session_state.dataset = response["dataset_id"]
                        st.success("Dataset uploaded successfully!")
                    else:
                        st.error(f"Error uploading dataset: {response['message']}")

            except Exception as e:
                st.error(f"Error processing uploaded file: {str(e)}")


def show_model_page():
    st.title("Model Selection")

    # Check if dataset is selected
    if "dataset" not in st.session_state:
        st.warning("Please select a dataset first.")
        return

    try:
        # Get available models from the API
        models = api_client.get_available_models()

        # Group models by type
        model_types = {}
        for model in models:
            if model["type"] not in model_types:
                model_types[model["type"]] = []
            model_types[model["type"]].append(model)

        # Model selection
        model_type = st.selectbox("Select model type", list(model_types.keys()))

        if model_type:
            model_options = [model["name"] for model in model_types[model_type]]
            selected_model = st.selectbox("Select a model", model_options)

            if selected_model:
                # Get model details
                model_details = next(
                    (
                        model
                        for model in model_types[model_type]
                        if model["name"] == selected_model
                    ),
                    None,
                )

                if model_details:
                    st.subheader("Model Information")
                    st.write(f"Name: {model_details['name']}")
                    st.write(f"Description: {model_details['description']}")

                    # Model hyperparameters
                    st.subheader("Model Hyperparameters")
                    hyperparams = {}

                    for param in model_details["parameters"]:
                        param_name = param["name"]
                        param_type = param["type"]
                        param_default = param["default"]
                        param_description = param.get("description", "")

                        st.write(f"**{param_name}**: {param_description}")

                        if param_type == "float":
                            hyperparams[param_name] = st.slider(
                                f"{param_name}",
                                min_value=param.get("min", 0.0),
                                max_value=param.get("max", 1.0),
                                value=param_default,
                                step=param.get("step", 0.01),
                            )
                        elif param_type == "int":
                            hyperparams[param_name] = st.slider(
                                f"{param_name}",
                                min_value=param.get("min", 1),
                                max_value=param.get("max", 100),
                                value=param_default,
                                step=param.get("step", 1),
                            )
                        elif param_type == "bool":
                            hyperparams[param_name] = st.checkbox(
                                f"{param_name}", value=param_default
                            )
                        elif param_type == "select":
                            hyperparams[param_name] = st.selectbox(
                                f"{param_name}",
                                options=param.get("options", []),
                                index=(
                                    param.get("options", []).index(param_default)
                                    if param_default in param.get("options", [])
                                    else 0
                                ),
                            )

                    # Save the selected model to session state
                    if st.button("Use this model"):
                        st.session_state.model = selected_model
                        st.session_state.model_type = model_type
                        st.session_state.hyperparams = hyperparams
                        st.success(f"Model '{selected_model}' selected!")

    except Exception as e:
        st.error(f"Error loading models: {str(e)}")


def show_noise_page():
    st.title("Noise Configuration")

    # Check if dataset and model are selected
    if "dataset" not in st.session_state:
        st.warning("Please select a dataset first.")
        return

    if "model" not in st.session_state:
        st.warning("Please select a model first.")
        return

    try:
        # Get available noise types from the API
        noise_types = api_client.get_available_noise_types()

        # Noise type selection
        selected_noise_type = st.selectbox("Select noise type", noise_types)

        if selected_noise_type:
            # Get noise type details
            noise_details = api_client.get_noise_type_details(selected_noise_type)

            st.subheader("Noise Information")
            st.write(f"Type: {noise_details['name']}")
            st.write(f"Description: {noise_details['description']}")

            # Noise parameters
            st.subheader("Noise Parameters")
            noise_params = {}

            for param in noise_details["parameters"]:
                param_name = param["name"]
                param_type = param["type"]
                param_default = param["default"]
                param_description = param.get("description", "")

                st.write(f"**{param_name}**: {param_description}")

                if param_type == "float":
                    noise_params[param_name] = st.slider(
                        f"{param_name}",
                        min_value=param.get("min", 0.0),
                        max_value=param.get("max", 1.0),
                        value=param_default,
                        step=param.get("step", 0.01),
                    )
                elif param_type == "int":
                    noise_params[param_name] = st.slider(
                        f"{param_name}",
                        min_value=param.get("min", 1),
                        max_value=param.get("max", 100),
                        value=param_default,
                        step=param.get("step", 1),
                    )
                elif param_type == "bool":
                    noise_params[param_name] = st.checkbox(
                        f"{param_name}", value=param_default
                    )
                elif param_type == "select":
                    noise_params[param_name] = st.selectbox(
                        f"{param_name}",
                        options=param.get("options", []),
                        index=(
                            param.get("options", []).index(param_default)
                            if param_default in param.get("options", [])
                            else 0
                        ),
                    )

            # Save the selected noise configuration to session state
            if st.button("Use this noise configuration"):
                st.session_state.noise_type = selected_noise_type
                st.session_state.noise_params = noise_params
                st.success(f"Noise configuration '{selected_noise_type}' selected!")

    except Exception as e:
        st.error(f"Error loading noise types: {str(e)}")


def show_training_page():
    st.title("Model Training")

    # Check if all required configurations are selected
    if "dataset" not in st.session_state:
        st.warning("Please select a dataset first.")
        return

    if "model" not in st.session_state:
        st.warning("Please select a model first.")
        return

    if "noise_type" not in st.session_state:
        st.warning("Please configure noise first.")
        return

    # Display selected configurations
    st.subheader("Selected Configurations")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Dataset**")
        st.info(st.session_state.dataset)

    with col2:
        st.write("**Model**")
        st.info(f"{st.session_state.model} ({st.session_state.model_type})")

    with col3:
        st.write("**Noise Type**")
        st.info(st.session_state.noise_type)

    # Training parameters
    st.subheader("Training Parameters")

    col1, col2 = st.columns(2)

    with col1:
        test_size = st.slider(
            "Test Size", min_value=0.1, max_value=0.5, value=0.2, step=0.05
        )
        random_state = st.number_input(
            "Random State", min_value=0, max_value=100, value=42
        )

    with col2:
        cv_folds = st.slider(
            "Cross-Validation Folds", min_value=2, max_value=10, value=5, step=1
        )
        experiment_name = st.text_input(
            "Experiment Name",
            value=f"{st.session_state.model}_{st.session_state.noise_type}",
        )

    # Start training
    if st.button("Start Training"):
        try:
            # Prepare training configuration
            training_config = {
                "dataset_id": st.session_state.dataset,
                "model_name": st.session_state.model,
                "model_type": st.session_state.model_type,
                "model_params": st.session_state.hyperparams,
                "noise_type": st.session_state.noise_type,
                "noise_params": st.session_state.noise_params,
                "test_size": test_size,
                "random_state": random_state,
                "cv_folds": cv_folds,
                "experiment_name": experiment_name,
            }

            # Start training
            with st.spinner("Training in progress..."):
                response = api_client.start_training(training_config)

                if response["success"]:
                    st.session_state.experiment_id = response["experiment_id"]
                    st.success(
                        f"Training started successfully! Experiment ID: {response['experiment_id']}"
                    )

                    # Display training progress
                    st.subheader("Training Progress")
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    import time

                    # Poll for training status
                    while True:
                        status = api_client.get_training_status(
                            response["experiment_id"]
                        )

                        if status["status"] == "completed":
                            progress_bar.progress(100)
                            status_text.success("Training completed successfully!")
                            break
                        elif status["status"] == "failed":
                            progress_bar.progress(100)
                            status_text.error(f"Training failed: {status['message']}")
                            break
                        else:
                            progress_bar.progress(status["progress"])
                            status_text.info(f"Status: {status['message']}")
                            time.sleep(1)

                    # Redirect to results page
                    if status["status"] == "completed":
                        st.session_state.page = "Results"
                        st.experimental_rerun()
                else:
                    st.error(f"Error starting training: {response['message']}")

        except Exception as e:
            st.error(f"Error during training: {str(e)}")


def show_results_page():
    st.title("Results Analysis")

    # Check if an experiment has been run
    if "experiment_id" not in st.session_state:
        # Allow selecting from previous experiments
        st.subheader("Select an Experiment")

        try:
            experiments = api_client.get_experiments()

            if experiments:
                selected_experiment = st.selectbox(
                    "Select an experiment",
                    options=[exp["id"] for exp in experiments],
                    format_func=lambda x: next(
                        (exp["name"] for exp in experiments if exp["id"] == x), x
                    ),
                )

                if selected_experiment:
                    st.session_state.experiment_id = selected_experiment
            else:
                st.warning("No experiments found. Please run a training first.")
                return

        except Exception as e:
            st.error(f"Error loading experiments: {str(e)}")
            return

    # Get experiment results
    try:
        experiment_results = api_client.get_experiment_results(
            st.session_state.experiment_id
        )

        # Display experiment information
        st.subheader("Experiment Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Experiment Name**")
            st.info(experiment_results["name"])

            st.write("**Dataset**")
            st.info(experiment_results["dataset"])

        with col2:
            st.write("**Model**")
            st.info(experiment_results["model"])

            st.write("**Noise Type**")
            st.info(experiment_results["noise_type"])

        with col3:
            st.write("**Noise Level**")
            st.info(f"{experiment_results['noise_level']}")

            st.write("**Date**")
            st.info(experiment_results["date"])

        # Display metrics
        st.subheader("Performance Metrics")

        metrics = experiment_results["metrics"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")

        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")

        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")

        with col4:
            st.metric("F1 Score", f"{metrics['f1']:.4f}")

        # Display visualizations
        st.subheader("Visualizations")

        # Tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(
            ["Confusion Matrix", "ROC Curve", "Learning Curve", "Feature Importance"]
        )

        with tab1:
            cm_url = experiment_results["visualizations"]["confusion_matrix"]
            st.write("Confusion Matrix URL:", cm_url)  # Добавьте эту строку
            st.image(
                f"mlruns/d33baa551c6e4b86afd8da4674029f37/artifacts/confusion_matrix.png"
            )

        with tab2:
            st.image(
                experiment_results["visualizations"]["roc_curve"], use_column_width=True
            )

        with tab3:
            st.image(
                experiment_results["visualizations"]["learning_curve"],
                use_column_width=True,
            )

        with tab4:
            st.image(
                experiment_results["visualizations"]["feature_importance"],
                use_column_width=True,
            )

        # Comparison with baseline (no noise)
        st.subheader("Comparison with Baseline (No Noise)")

        baseline_comparison = experiment_results["baseline_comparison"]

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Accuracy",
                f"{metrics['accuracy']:.4f}",
                f"{metrics['accuracy'] - baseline_comparison['accuracy']:.4f}",
                delta_color="inverse",
            )

        with col2:
            st.metric(
                "Precision",
                f"{metrics['precision']:.4f}",
                f"{metrics['precision'] - baseline_comparison['precision']:.4f}",
                delta_color="inverse",
            )

        with col3:
            st.metric(
                "Recall",
                f"{metrics['recall']:.4f}",
                f"{metrics['recall'] - baseline_comparison['recall']:.4f}",
                delta_color="inverse",
            )

        with col4:
            st.metric(
                "F1 Score",
                f"{metrics['f1']:.4f}",
                f"{metrics['f1'] - baseline_comparison['f1']:.4f}",
                delta_color="inverse",
            )

        # Noise impact analysis
        st.subheader("Noise Impact Analysis")

        st.write(experiment_results["noise_impact_analysis"])

        # Download results
        st.subheader("Download Results")

        col1, col2 = st.columns(2)

        with col1:
            st.download_button(
                "Download Report (PDF)",
                experiment_results["report_pdf"],
                file_name=f"experiment_{st.session_state.experiment_id}_report.pdf",
                mime="application/pdf",
            )

        with col2:
            st.download_button(
                "Download Data (CSV)",
                experiment_results["results_csv"],
                file_name=f"experiment_{st.session_state.experiment_id}_data.csv",
                mime="text/csv",
            )

    except Exception as e:
        st.error(f"Error loading experiment results: {str(e)}")


if __name__ == "__main__":
    main()

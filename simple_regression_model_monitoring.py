import numpy as np
import pandas as pd
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import RegressionPreset
from evidently.ui.workspace import Workspace

def get_data():
    """
    Loads the diabetes dataset from sklearn and creates a DataFrame.
    """
    diabetes_data = datasets.load_diabetes()
    diabetes_df = pd.DataFrame(diabetes_data.data, columns=diabetes_data.feature_names)
    print(f"df data sample:\n{diabetes_df.head()}")
    print(f"df shape: {diabetes_df.shape}")
    diabetes_df['target'] = diabetes_data.target
    print(f"Target df shape: {diabetes_df['target'].shape}")
    diabetes_df['prediction'] = diabetes_df['target'].values + np.random.normal(0, 3, diabetes_df.shape[0])
    print(f"Prediction df shape: {diabetes_df['prediction'].shape}")
    print(f"df data sample:\n{diabetes_df.head()}")
    return diabetes_df

def prepare_data(dataframe, ref_sample_size, curr_sample_size):
    """
    Prepares reference and current datasets for comparison.
    """
    reference_data = dataframe.sample(n=ref_sample_size, replace=False)
    print(f"Reference data shape: {reference_data.shape}")
    current_data = dataframe.sample(n=curr_sample_size, replace=False)
    print(f"Current data shape: {reference_data.shape}")    
    return reference_data, current_data

def generate_regression_performance_report(reference_data, current_data):
    """
    Generates a regression performance report using Evidently.
    """
    # TODO : Create a report instance for regression with RegressionPreset()

    # Initialize the report with desired metrics
    regression_performance_report = Report(metrics=[
        RegressionPreset()
    ])

    # Run the report
    regression_performance_report.run(reference_data=reference_data.sort_index(), current_data=current_data.sort_index())
    return regression_performance_report

def add_report_to_workspace(workspace, project_name, project_description, report):
    """
    Adds a report to an existing or new project in a workspace.
    This function will be useful to you
    """
    # Check if project already exists
    project = None
    for p in workspace.list_projects():
        if p.name == project_name:
            project = p
            break

    # Create a new project if it doesn't exist
    if project is None:
        project = workspace.create_project(project_name)
        project.description = project_description

    # Add report to the project
    workspace.add_report(project.id, report)
    print(f"New report added to project {project_name}")

if __name__ == "__main__":
    WORKSPACE_NAME = "datascientest-workspace"
    PROJECT_NAME = "simple_regression_model_monitoring"
    PROJECT_DESCRIPTION = "Evidently Dashboards"

    # Get the diabetes dataset
    diabetes_df = get_data()

    # Prepare reference and current datasets by taking a sample of 50 rows
    reference_data, current_data = prepare_data(diabetes_df, ref_sample_size=50, curr_sample_size=50)

    # Generate the regression performance report
    regression_report = generate_regression_performance_report(reference_data, current_data)

    # Create and Add report to workspace
    workspace = Workspace.create(WORKSPACE_NAME)
    add_report_to_workspace(workspace, PROJECT_NAME, PROJECT_DESCRIPTION, regression_report)

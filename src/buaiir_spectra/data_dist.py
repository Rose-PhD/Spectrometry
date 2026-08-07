                                                                                                
from huggingface_hub import login, upload_folder
from huggingface_hub import HfApi, CommitOperationDelete
from buaiir_spectra.utils.config_info import HF_TOKEN

login()

api = HfApi()

# api.upload_large_folder(
#     folder_path="src/buaiir_spectra/data_splits", 
#     repo_id="wilfredk/labeled_data", 
#     repo_type="dataset"
# )

# api.create_commit(
#     repo_id="wilfredk/labeled_data",
#     repo_type="dataset",
#     operations=[
#         CommitOperationDelete(path_in_repo="BIO_SCIENCE"),
#         # CommitOperationDelete(path_in_repo="SCAN_CODER"),
#         CommitOperationDelete(path_in_repo="SCAN_CORDER"),
#         CommitOperationDelete(path_in_repo="LOW_COST")
#     ],
#     commit_message="Wipe old root directories before pushing corrected files"
# )

# # 2. Upload cleanly into a dedicated "data_splits" subfolder
# api.upload_large_folder(
#     folder_path="src/buaiir_spectra/data_splits", 
#     repo_id="wilfredk/labeled_data", 
#     repo_type="dataset",
#     repo_path="data_splits"  # <--- Correct argument name for upload_large_folder
# )



# login()

# upload_folder(folder_path=".", repo_id="wilfredk/raw_dataset", repo_type="dataset")

api.upload_large_folder(
    folder_path="/home/wilfred/Downloads/spectra_data_clean", 
    repo_id="wilfredk/raw_dataset", 
    repo_type="dataset"
)

# data loading scri
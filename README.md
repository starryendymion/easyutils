# Easyutils

**Disclaimer:** Easyutils is a lightweight Python module designed for non-profit and personal use. It leverages scikit-learn for streamlined feature engineering. All rights to scikit-learn and other libraries used belong to their respective owners.

<span style="color:green">**Important info:** `demonstration.ipynb` is a notebook file which you can view. It shows how this module can be used on a structured dataset with the most efficiency.</span>

## Classes

### Dataframe Inspector

**Methods:**

#### inspect(dataframe)

- **Dataframe:** The pandas DataFrame to be inspected.
- **Functionality:** Provides key insights into the DataFrame, including NaN values, duplicate rows, constant columns, and those with high missing values.

#### retrieve_column_types(dataframe)

- **Dataframe:** The pandas DataFrame for column categorization.
- **Functionality:** Categorizes columns into numeric, object, boolean, and miscellaneous types, returning lists of column names for each category.

#### retrieve_column_info(dataframe, exclude=None)

- **Dataframe:** The pandas DataFrame for column information.
- **Exclude:** List of columns to be excluded (optional).
- **Functionality:** Generates detailed information for each column based on its storage data type, excluding specified columns if provided.

#### retrieve_labels(dataframe, columns_to_extract)

- **Dataframe:** The pandas DataFrame for label extraction.
- **Columns_to_extract:** List of columns to be extracted as labels.
- **Functionality:** Extracts specified columns, converts them to numpy arrays, and returns them. Also drops these columns from the dataframe.

### Data Preprocessor

**Methods:**

#### apply_and_save(data_array, method, save_location_pkl)

- **Data_array:** The data array to be preprocessed.
- **Method:** The preprocessing method to be applied. Choose from "OneHotEncoder," "StandardScaler," "RobustScaler," or "MinMaxScaler."
- **Save_location_pkl:** The file path to save the fitted preprocessing object (pickle file).
- **Functionality:** Applies the specified preprocessing method to the data array, saves the fitted preprocessing object, and returns the transformed data.

#### load_and_apply(object_location, data_array)

- **Object_location:** The file path to load the fitted preprocessing object (pickle file).
- **Data_array:** The data array to be transformed using the loaded preprocessing object.
- **Functionality:** Loads the fitted preprocessing object, applies it to the data array, and returns the transformed data.

#### assign_binary(data_array, zero_class, one_class)

- **Data_array:** The data array to be transformed into binary values.
- **Zero_class:** The value to be mapped to 0.
- **One_class:** The value to be mapped to 1.
- **Functionality:** Maps specified classes to binary values (0 or 1) in the data array and returns the transformed data.

### Data Decomposer

**Methods:**

#### observe(data_array, method, check=None)

- **Data_array:** The data array to be observed.
- **Method:** The decomposition method to be used, choose from "pca" or "tsvd."
- **Check:** Number of components to check explained variance (optional).
- **Functionality:** Observes the data using PCA or truncated singular value decomposition (tsvd) and visualizes cumulative explained variance. Can check explained variance with a specific number of components if specified.

#### apply_and_save(data_array, method, num_components, save_location_pkl)

- **Data_array:** The data array to be decomposed.
- **Method:** The decomposition method to be applied, choose from "pca" or "tsvd."
- **Num_components:** The number of components to retain during decomposition.
- **Save_location_pkl:** The file path to save the fitted decomposition object (pickle file).
- **Functionality:** Applies PCA or tsvd to the data array, saves the fitted decomposition object, and returns the transformed data.

#### load_and_apply(object_location, data_array, method)

- **Object_location:** The file path to load the fitted decomposition object (pickle file).
- **Data_array:** The data array to be transformed using the loaded decomposition object.
- **Method:** The decomposition method used, choose from "pca" or "tsvd."
- **Functionality:** Loads the fitted decomposition object and applies it to the data array, returning the transformed data.

#### tsvd_view_decomposed_image(object_location, image_array, width, height)

- **Object_location:** The file path to load the fitted tsvd decomposition object (pickle file).
- **Image_array:** The flattened image array to be visualized.
- **Width:** The width of the original image.
- **Height:** The height of the original image.
- **Functionality:** Reconstructs and visualizes images from their truncated singular value decomposition.

**Note:** Easyutils is designed for structured datasets and aims to simplify common feature engineering tasks. It enhances convenience and efficiency in the preprocessing and decomposition of data.

**Important:** This module is intended for non-profit and personal use only. All credit for scikit-learn and other utilized libraries goes to their respective owners.

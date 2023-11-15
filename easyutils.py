import pandas as pd
import numpy as np
import joblib
import math

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

from sklearn.preprocessing import OneHotEncoder, StandardScaler,RobustScaler,MinMaxScaler
from sklearn.decomposition import PCA,TruncatedSVD




#--------------------------Dataframe Inspection------------------------------------------

class Dataframe_Inspector:
    def inspect(self, df):
        nan_values = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        constant_columns = (df.nunique() == 1).sum()
        high_missing_columns = (df.isnull().mean() > 0.5).sum()
        
        print("NaN Values:", nan_values)
        print("Duplicate Rows:", duplicate_rows)
        print("Constant Columns:", constant_columns)
        print("High Missing Value Columns:", high_missing_columns)

    def retrieve_column_types(self, df):
        numeric_cols = []
        object_cols = []
        boolean_cols = []
        misc_cols = []
        
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                if df[col].nunique() == 2:
                    boolean_cols.append(col)
                else:
                    object_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].nunique() == 2:
                    boolean_cols.append(col)
                else:
                    numeric_cols.append(col)
            else:
                misc_cols.append(col)

        return numeric_cols, object_cols, boolean_cols, misc_cols

    def retrieve_column_info(self, df, exclude=None):
        if exclude is not None:
            numeric_cols, object_cols, boolean_cols, misc_cols = self.retrieve_column_types(df.drop(exclude, axis=1))
        else:
            numeric_cols, object_cols, boolean_cols, misc_cols = self.retrieve_column_types(df)

        numeric_data = "Numeric Columns:\n"
        for col in numeric_cols:
            numeric_data += f"  Column: {col}\n"
            numeric_data += f"  Datatype: {df[col].dtype}\n"
            numeric_data += f"  Min: {df[col].min()}\n"
            numeric_data += f"  Max: {df[col].max()}\n"
            numeric_data += f"  Mean: {df[col].mean()}\n"
            numeric_data += f"  Median: {df[col].median()}\n"
            numeric_data += f"  Mode: {df[col].mode()[0]}\n\n"

        object_data = "Object Columns:\n"
        for col in object_cols:
            object_data += f"  Column: {col}\n"
            object_data += f"  Datatype: {df[col].dtype}\n"
            object_data += f"  Unique Values: {df[col].unique()}\n\n"

        boolean_data = "Boolean Columns:\n"
        for col in boolean_cols:
            boolean_data += f"  Column: {col}\n"
            boolean_data += f"  Datatype: Boolean\n"
            boolean_data += f"  Unique Values: {df[col].unique()}\n\n"

        misc_data = "Miscellaneous Columns:\n"
        for col in misc_cols:
            misc_data += f"  Column: {col}\n"
            misc_data += f"  Datatype: {df[col].dtype}\n\n"

        return numeric_data, object_data, boolean_data, misc_data
    
    def retrieve_labels(self,df, columns_to_extract):
       labels = df[columns_to_extract].to_numpy()
       df.drop(columns=columns_to_extract, inplace=True)
       return labels

    
#----------------------Simple Preprocessing Techniques-------------------------------------

class Data_Preprocessor:
    def __init__(self):
        self.preprocessors = {
            "OneHotEncoder": OneHotEncoder(sparse_output=False),
            "StandardScaler": StandardScaler(),
            "RobustScaler": RobustScaler(),
            "MinMaxScaler": MinMaxScaler()
        }

    def apply_and_save(self, method, data_array, save_location_pkl):
        if method not in self.preprocessors:
            raise ValueError("Invalid preprocessing method choice. Availiable Methods : ",self.preprocessors.keys())

        pkl_object = self.preprocessors[method]
        transformed_data = pkl_object.fit_transform(np.array(data_array))
        joblib.dump(pkl_object, save_location_pkl)

        if method == "OneHotEncoder":
            print("OneHotEncoder Categories: \n", pkl_object.categories_)
            print("\n")

        print(f"{pkl_object.__class__.__name__} dumped to: {save_location_pkl}\n")

        return transformed_data

    def load_and_apply(self, object_location, data_array):
        pkl_object = joblib.load(object_location)
        return pkl_object.transform(data_array)
    
    
    def assign_binary(self, data_array,zero_class,one_class):
     class_mapping = {zero_class: 0,one_class:1}
     binary_data = np.vectorize(class_mapping.get)(data_array)
     return binary_data

#-------------------------Dimensionality Reduction Techniques------------------------

class Data_Decomposer:
    def __init__(self):
        pass

    def _change_image(self, image_array):
        num_images = image_array.shape[0]
        flattened_shape = np.prod(np.array(image_array.shape[1:]))
        changed_array = image_array.reshape(num_images, flattened_shape)
        return changed_array, flattened_shape

    def observe(self, data_array, method,check=None):

        if method not in ["pca", "tsvd"]:
          raise ValueError("Please choose method: 'pca' or 'tsvd'")
        

        if method == "pca":
            pca_all = PCA()
            pca_all.fit(data_array)
            cum_var = np.cumsum(pca_all.explained_variance_ratio_)
            n_comp = np.arange(1, pca_all.n_components_ + 1)

        elif method == "tsvd":
            image_array, flattened_shape = self._change_image(data_array)
            tsvd = TruncatedSVD(n_components=flattened_shape - 1)
            tsvd.fit(image_array)
            cum_var = np.cumsum(tsvd.explained_variance_ratio_)
            n_comp = np.arange(1, flattened_shape)

        ax = sns.pointplot(x=n_comp, y=cum_var)
        ax.set(xlabel="Number of Components", ylabel="Cumulative Explained Variance")
        ax.xaxis.set_major_locator(MultipleLocator(base=math.ceil(np.prod(np.array(data_array.shape[1:]))/10)))
        plt.show()

        if check !=None:
         print(f"Explained variance with {check} components: {float(cum_var[check-1:check])*100:.2f}%")

    
    def apply_and_save(self, data_array, method, num_components, save_location_pkl):

        if method not in ["pca", "tsvd"]:
          raise ValueError("Please choose method: 'pca' or 'tsvd'")

        if method == "pca":
            pca_n = PCA(n_components=num_components)
            transformed_data = pca_n.fit_transform(data_array)
            joblib.dump(pca_n, save_location_pkl)
            print("PCA object dumped to:", save_location_pkl)
            return transformed_data
        
        elif method == "tsvd":
            image_array, flattened_shape = self._change_image(data_array)
            tsvd = TruncatedSVD(n_components=num_components)
            transformed_images = tsvd.fit_transform(image_array)
            joblib.dump(tsvd, save_location_pkl)
            print("tsvd object dumped to:", save_location_pkl)
            return transformed_images
        
    def load_and_apply(self, object_location, data_array, method):
     pkl_object = joblib.load(object_location)

     if method not in ["pca", "tsvd"]:
        raise ValueError("Please choose method: 'pca' or 'tsvd'")

     if method == "pca":
        transformed_output = pkl_object.transform(data_array)
     elif method == "tsvd":
        image_array, flattened_shape = self._change_image(data_array)
        transformed_output = pkl_object.transform(image_array)

     return transformed_output
    
    def tsvd_view_decomposed_image(self,object_location,image_array,width,height):
        pkl_object = joblib.load(object_location)
        image_reduced=pkl_object.inverse_transform(image_array.reshape(1,-1))
        image_reduced=image_reduced.reshape((width,height))
        plt.matshow(image_reduced,cmap='gray')
        plt.show()

 
 



    

     

    








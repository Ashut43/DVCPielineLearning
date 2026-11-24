import pandas as pd
import os 
from pathlib import Path 
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
def create_dataframe():
    data= {
            'Id':[1,2,3,4,5,6,7,8,9,10],
            "reviews":[
                        "Inception is a mind-bending masterpiece.",
                        "The Office never fails to make me laugh.",
                        "Interstellar left me emotionally devastated.",
                        "Breaking Bad has perfect character development.",
                        "Parasite is a brilliant social thriller.",
                        "Stranger Things captures pure 80s nostalgia.",
                        "The Dark Knight redefined superhero films.",
                        "Ted Lasso is heartwarming and genuinely funny.",
                        "Whiplash delivers intense, unforgettable performances.",
                        "Game of Thrones started strong, ended poorly "]
    }
    df=pd.DataFrame(data)
    return df

def save_dataFrame(df):
    if not os.path.exists("Data"):
        os.mkdir("Data")
    df.to_csv("Data/data.csv",index=False)
    print("data.csv file Saved Successfully")
def process_data(k):
    df=pd.read_csv("Data/data.csv")
    Cv=CountVectorizer(max_features=k)
    Cv_Data=Cv.fit_transform(df["reviews"]) 
    Cv_feature=Cv.get_feature_names_out()
    Vectarized_df=pd.DataFrame(Cv_Data.toarray(),columns=Cv_feature)
    processed_df=pd.concat([df,Vectarized_df])
    processed_df.to_csv('Data/processes_data.csv',index=False)
    print(f"processed has been saved in Data folder with {k} columns")
    return processed_df

if __name__=='__main__':
    df=create_dataframe()
    save_dataFrame(df)
    k=3
    ProcessData=process_data(k)
    print(f"data sahpe:{df.shape}")
    print(f"processed data sahpe:{ProcessData.shape}")
    





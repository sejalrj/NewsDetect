import streamlit as st
import pandas as pd
from streamlit_ace import st_ace
from PIL import Image
from pyspark.ml.feature import CountVectorizerModel, Tokenizer
from pyspark.ml.classification import RandomForestClassificationModel
import pydeck as pdk
import numpy as np

import os
import pyspark

# Define a sample dataframe
def load_data(sqlc, file):
    new_data = sqlc.read.option("multiline",True)\
              .option("mode", "DROPMALFORMED")\
              .option("dateFormat", "yyyy-MM-dd")\
              .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")\
              .csv(file, header=True)
    
    new_data = new_data.limit(100)

    new_data = new_data.select('content')
    new_data = new_data.dropna()

    t = Tokenizer(inputCol='content', outputCol='words')
    td = t.transform(new_data)
    return td

# Add a section for the image collage
def display_images():
    images = [Image.open('obi-pixel7propix-UEQvUtRs224-unsplash.jpeg'), Image.open("5FR.gif")]

    image = Image.open('obi-pixel7propix-UEQvUtRs224-unsplash.jpeg')
    width, height = image.size
    num_images = len(images)
    image_width = int(width / num_images)
    image_height = height

    new_image = Image.new('RGB', (width, height), (255, 255, 255))
    x_offset = 0
    for i in range(num_images):
        resized_image = images[i].resize((image_width, image_height))
        new_image.paste(resized_image, (x_offset, 0))
        x_offset += image_width
    st.image(new_image, caption='News - Oh! is it fake or real? Is it heavily influenced by political decisions? Or will it influence political decisions heavily? 🤯')

# Add a section for the area chart
def display_area_chart(df):
    st.area_chart(df)

# Add a section for the table
# def display_table(df):
#     st.header("📰 News and Trends")
#     st.dataframe(df)

# Add a section for the text input
def display_text_input(sqlc, loadedcvmodel, rfcmodel):
    st.header("👀 Enter News Text")
    news_text = st.text_input("🤔 Enter the news text here", "")
    news_df = sqlc.createDataFrame([(news_text,)], ["content"])

    news_df = news_df.select('content')
    news_df = news_df.dropna()

    t = Tokenizer(inputCol='content', outputCol='words')
    td = t.transform(news_df)
    news_data = loadedcvmodel.transform(td)
    print("YES4")
    # df = sqlc.read.json('/Users/sejalrameshjagtap/Desktop/vectorized/part-00000-419610a2-c7e3-4f97-a47a-7f2d14b49039-c000.json')
    predictions = rfcmodel.transform(news_data)
    news_text2 = ""
    news_text2 = predictions.select('prediction').collect()

    mapping= {'0.0': 'satire',
'1.0': 'political',
'2.0': 'unreliable',
'3.0': 'bias',
'4.0': 'conspiracy',
'5.0': 'clickbait',
'6.0': 'reliable',
'7.0': 'hate',
'8.0': 'rumor',
'9.0': 'junksci',
'10.0': 'fake',
'11.0': 'unknown'}
    st.write("🔍 This news is potentially", mapping.get(str(news_text2[0][0])))

def get_spark_context():
    
    conf = pyspark.SparkConf()
    conf.set("spark.driver.memory", "8g")
    conf.set("spark.worker.timeout", "10000000")
    conf.set("spark.driver.maxResultSize", "0")
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.extraClassPath", "/usr/local/cuda-11.2/lib64") \
                    .set("spark.executor.extraClassPath", "/usr/local/cuda-11.2/lib64") \
                    .set("spark.driver.extraLibraryPath", "/usr/local/cuda-11.2/lib64") \
                    .set("spark.executor.extraLibraryPath", "/usr/local/cuda-11.2/lib64") \
                    .set("spark.rapids.sql.concurrentGpuTasks", "") \
                    .set("spark.rapids.sql.variableFloatAgg.enabled", "true")
    spark = pyspark.SparkContext(conf=conf)
    conf.set('spark.sql.repl.eagerEval.enabled', True)

    sqlc = pyspark.SQLContext.getOrCreate(spark)
    return sqlc
    
def display_table(df):
    st.header("📰 News and Trends")
    


    # mapping = {'fake': 10.0, 'conspiracy': 4.0, 'unreliable': 2.0, 'political': 1.0, 'hate': 7.0, 'reliable': 6.0, 'satire': 0.0, 'clickbait': 5.0, 'junksci': 9.0, 'bias': 3.0, 'rumor': 8.0, 'unknown': 11.0}
    mapping= {'0.0': 'satire',
'1.0': 'political',
'2.0': 'unreliable',
'3.0': 'bias',
'4.0': 'conspiracy',
'5.0': 'clickbait',
'6.0': 'reliable',
'7.0': 'hate',
'8.0': 'rumor',
'9.0': 'junksci',
'10.0': 'fake',
'11.0': 'unknown'}
    df = df.rename(columns={'_c1': 'News', '_c2': 'prediction'})

    # dropping the _c0 column
    df = df.drop('_c0', axis=1)
    
    
    df['prediction'] = df['prediction'].map(mapping)

    df = df[df['News'].str.contains('San Francisco|SF')]
    
    num_rows = df.shape[0]
    page_size = 25 # Number of rows to display per page
    page_count = num_rows // page_size + 1
    page_number = st.slider("Page", 1, page_count, 1)
    start_idx = (page_number - 1) * page_size
    end_idx = min(start_idx + page_size, num_rows)
    st.write(df[start_idx:end_idx])


@st.cache_resource()
def init_spark():
    
    
    # Create a Spark session
    # Load the saved model
    sqlc = get_spark_context()

    # Tokenize text data
    loadedcvmodel = CountVectorizerModel.load("/Users/sejalrameshjagtap/Desktop/temp/my_vectorizer_model3")

    # Load the trained model from disk
    rfcmodel = RandomForestClassificationModel.load("/Users/sejalrameshjagtap/Desktop/temp/RFC_small5_Model.mdl")

    #TO RUN THE BIG MODEL
    # rfcmodel = RandomForestClassificationModel.load("/Users/sejalrameshjagtap/Desktop/RFCModel.mdl")

    return sqlc, loadedcvmodel, rfcmodel

def display_map():
    chart_data = pd.DataFrame(np.random.randn(100, 2) / [100, 100] + [37.76, -122.4],
                          columns=['lat', 'lon'])
        # Define the number of random places to generate
        
    st.pydeck_chart(pdk.Deck(
        map_style=None,
        initial_view_state=pdk.ViewState(
            latitude=37.76,
            longitude=-122.4,
            zoom=11,
            pitch=50,
        ),
        layers=[
            pdk.Layer(
            'HexagonLayer',
            data=chart_data,
            get_position='[lon, lat]',
            radius=200,
            elevation_scale=4,
            elevation_range=[0, 1000],
            pickable=True,
            extruded=True,
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=chart_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius=200,
            ),
        ],
    ))

def main():
    st.set_page_config(page_title="News Trends", page_icon=None, layout="wide")
    print("YES")
    sqlc, loadedcvmodel, rfcmodel = init_spark()
    print("YES2")
    td = load_data(sqlc, "news_sample.csv")
    print("YES3")
    new_data = loadedcvmodel.transform(td)
    print("YES4")
    # df = sqlc.read.json('/Users/sejalrameshjagtap/Desktop/temp/vectorized/part-00000-419610a2-c7e3-4f97-a47a-7f2d14b49039-c000.json')
    predictions = rfcmodel.transform(new_data)
    print("predictions:",predictions)
    
    # pandas_df = predictions.select('content', 'prediction').toPandas()
    pandas_df = sqlc.read.option("multiline",True).option("mode", "DROPMALFORMED").csv("/Users/sejalrameshjagtap/Desktop/temp/predictions_f/part-00000-06329a45-648b-4df7-b48c-f1ed06fea5fa-c000.csv").toPandas()
    
    print("YES5")
    display_images()

    # display_area_chart(pandas_df)

    # Divide the page into two columns


    # Add a section for the table in the left column
    
    display_table(pandas_df)
    if st.button("Create Map"):
        display_map()

    # Add a section for the text input in the right column
    display_text_input(sqlc, loadedcvmodel, rfcmodel)


if __name__ == '__main__':
    main()

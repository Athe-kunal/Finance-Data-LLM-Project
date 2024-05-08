## INSTRUCTIONS TO RUN

1. Start the docker container for qdrant by running (See more instructions [here](https://qdrant.tech/documentation/guides/installation/#docker-and-docker-compose))
```
docker run -p 6333:6333 \
    -v $(pwd)/path/to/data:/qdrant/storage \
    qdrant/qdrant
```

2. Run the requirements file for installing the packages

3. Run the streamlit file for the demo

```
streamlit run Intro.py
```



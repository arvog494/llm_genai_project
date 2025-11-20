To get data to place in data/raw: 

  -Git clone the following repo https://github.com/vectara/open-rag-bench
  -create a virtual environment
  -install modules from requirements.txt
  -run the python file get_arvix.py to retrieve the data ( takes a long time because there is a lot .pdf to retrieve)
  - move files in directory data/raw of this current repo ( llm_genai_project)
  - if you want to do a quick test/check , you can only move a small number of files (<100 files)

The get_arvix.py file is not correct , you will get errors running it but you will quckly be able to correct this file
it is due to absolute path and putting the retrieved files in a directory that does not exist yet

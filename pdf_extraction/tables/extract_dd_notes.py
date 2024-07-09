#https://github.com/deepdoctection/notebooks/blob/main/Get_Started.ipynb
#py -3.8 -m pip install torchvision timm

import cv2
from pathlib import Path
from matplotlib import pyplot as plt
from IPython.core.display import HTML

import deepdoctection as dd

image_path = "./tables/sample_2.png"
image = cv2.imread(image_path)
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
plt.show()

#analyzer = dd.get_dd_analyzer(config_overwrite=["LANGUAGE='deu'"])
analyzer = dd.get_dd_analyzer()

#path = Path.cwd() / "pics/samples/sample_2"
path = "./papers/21222096.pdf"

df = analyzer.analyze(path=path)
df.reset_state()  # This method must be called just before starting the iteration. It is part of the API.

page =[]

for doc in df: 
	page.append(doc)


doc=iter(df)
page = next(doc)

type(page)

print(f" height: {page.height} \n width: {page.width} \n file_name: {page.file_name} \n document_id: {page.document_id} \n image_id: {page.image_id}\n")


page.get_attribute_names()

page.document_type, page.language

image = page.viz()
plt.figure(figsize = (25,17))
plt.axis('off')
plt.imshow(image)
plt.show()

for layout in page.layouts:
    if layout.category_name=="title":
        print(f"Title: {layout.text}")

page.chunks[0]

table = page.tables[0]
table.get_attribute_names()

print(f" number of rows: {table.number_of_rows} \n number of columns: {table.number_of_columns}" )

table.csv
table.text

cell = table.cells[0]
cell.get_attribute_names()

word = cell.words[0]
word.get_attribute_names()

t1 = table.csv
t2 = pd.DataFrame(t1[1:],columns = t1[0])
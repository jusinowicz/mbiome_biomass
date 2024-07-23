
#==============================================================================
# This section contains miscellaneous tools for parsing and visualizing the 
# sentence dependency structures. This might be temporary
#==============================================================================
# Generate the dependency tree in a textual format
# Access the specific sentence using doc.sents
sentence_index = 6
sentence = list(results_doc.sents)[sentence_index]

# Function to recursively print the tree structure
def print_tree(token, indent=''):
	print(f"{indent}{token.text} <-- {token.dep_} <-- {token.head.text}")
	for child in token.children:
		print_tree(child, indent + '  ')

for token in sentence:
    print_tree(token)

# Save the syntacticdependency visualization to an HTML file
from spacy import displacy
html = displacy.render(sentence, style="dep", page=True)
with open("./output/syntactic_tree_ex4.html", "w", encoding="utf-8") as file:
    file.write(html)

# Generate the dependency tree in html
# displacy.render(doc, style="dep", options={"compact": True, "color": "blue"})
# tree = displacy.render(sentence, style="dep", options={"compact": True, "color": "blue"})


s1 = "The plant dry weight was improved with the application of Bradyrhizobium by 59.3, 13.5, and 34.8%; and with the application of AMF by 63.2, 21.8, and 41.0% and with their combination by 61.7, 18.7, 38.7% in both growing seasons as compare with control, 100% NPK and 50% NPK respectively(Table 2)."
s2 = "The applications of fertilizer and AMF increased the dry weight by 100 and 300%, respecticely."
s3 = "The application of fertilizer increased the dry weight by 100%, while the application of AMF increased the dry weight by 300%."
s4 = "The highest dry biomass shoot found was 10.39 g  and root 9.59 g/plant in T3 inoculated with AMF in  T. arjuna, which was 29.71% and 19.72% higher  compared to non-inoculated control plants grown in  the same ratio of soil (Table 2)."




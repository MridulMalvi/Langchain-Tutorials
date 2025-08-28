from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.output_parsers import *
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()
import os

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")
)
model1 = ChatHuggingFace(llm=llm)

model2 = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

prompt1= PromptTemplate(
    input_variables=["input"],
    template="generate short and simple notes from given text: {input}"
)

prompt2 = PromptTemplate(
    input_variables=["input"],
    template="generate 5 short Question and answers from given text: {input}"
)

prompt3 = PromptTemplate(
    input_variables=["notes", "quiz"],
    template="Merge both provided notes and quiz: {notes} and {quiz}"
)

parser = StrOutputParser()

parallel_chain = RunnableParallel( 
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)
merge_chain = prompt3 | model2 | parser

chain = parallel_chain | merge_chain

input_text ="""Data is all around us. There are many kinds of data sources, including financial, biological, and social data. Even this page has data! For example, it has a total word count, dates it was created and last revised, a topic and subject matter, a number of page views, and languages that the content is available in.

However, while everything is potentially a source of data, data that is not recorded and organized may as well not exist at all. Without an underlying structure, data appears meaningless and fails to provide useful information.

By organized, we mean categorized in a standard and unambiguous way. The organized and categorized data is what we refer to when we say structured data.


Wikidata features form-based input for adding data to items
Where is structure?
On the web, structure reigns. Most websites are created using HTML, a markup language which provides the basic scaffolding, or structure, of a web page.

Markup languages are also used for tagging and describing page content so that search engines, bots, and applications like RSS feeds can easily process and "understand" it. For example, <title> tags tell machines what the name of a website is.

Instead of supporting the structure and common elements of a web page, Wikidata provides structure for all the information stored in Wikipedia, and on the other Wikimedia projects. Wikidata is based on the Mediawiki software as is any other Wikimedia project, extended by Wikibase, the software which powers Wikidata and is designed to manage large amounts of structured data. Structure is not directly added to the content of Wikipedia or other Wikimedia site pages, as in tables or lists, nor is any knowledge of markup languages, data schemas, object notation, or other special syntax required by Wikidata users; instead, data is added to and edited in Wikidata through user-friendly input forms.

All data stored on Wikidata can be used to generate all kinds of automated and up to date lists or tables or other structured pages in any Wikimedia site or elsewhere.

Structuring data
For an example on the importance of structure, let's look at Table 1. In this table we can see data for the four highest mountains on Earth. If we would like to know a particular piece of information, such as the height of the second highest mountain in the world, we should be able to look at the provided data and find out the correct value. However, only three of the four mountains have their data categorized as a height value, and only two of those three mountains have values in metres. While we know that height and hauteur (French for height) can be understood as equal to each other, and how to convert metres to feet or vice versa, a machine, such as a bot or a computer program may not.

It would be much easier for both humans and machines to process the information and answer the original question about the second highest mountain when all underlying data is recorded in a similar way even if the presentation differs.

Modeling data
Collections of structured data, like Wikidata, are organized according to a data model. Data models are machine-readable, meaning they can be understood by a computer. While computers are powerful, they are often not as smart as us when it comes to simple reasoning. For instance, in the example above, a machine would not be able to know that height and hauteur are the same unless they were explicitly told somehow that was the case.

Data for Mountains
Mountain	Property	Value
Mount Everest	continent	Asia
K2	continent	Asia
Kanchenjunga	continent	Asia
Lhotse	continent	Asia

Data models vary based on the analysis needs, scope and conceptual framework of the dataset, and the technical requirements of a system. However, all data models typically will specify what kind of data can be supported by a system and what relationships between values can be understood and represented. For example, a data model could specify that height and hauteur be mapped to each other so that both terms represent one concept, or that measurements in feet be automatically converted into metres. The Wikidata data model shapes the way that data can be edited and added to the system by users. It is also a work in progress, with new data types being added to the model over time.

The data model also essentially translates human natural language patterns into something that can be processed by machines. For example, in English we might say:

"Mount Everest is the highest mountain in the world"
This is also the raw, unstructured format of content currently on Wikipedia and all other Wikimedia sites.

On Wikidata, this would be represented by a statement, which consists of a property-value pair about an item, in this case Earth:

Earth (Q2) (item) → highest point (P610) (property) → Mount Everest (Q513) (value)
Additionally, Wikidata would also hold a statement about the item for Mount Everest (indicating it is a mountain):

Mount Everest (Q513) (item) → instance of (P31) (property) → mountain (Q8502) (value)
Note that because other items can be used as the values for statements, and all items have their own unique page on Wikidata, this means that all items in the system can be linked together through a series of statements. Because Wikidata uses a machine-readable format, this interlinking of data allows new relationships and connections to be discovered and processed by machines. For example, in Table 2 we see new data for our mountains, this time about their geographical location by continent but nothing about their heights. Assuming this continent data was linked to the mountain height data, we would feel more confident making predictions or drawing certain conclusions about it, like saying that Asia is home to the world's highest mountains.


Linking data
Besides being a collection of structured data, Wikidata also supports linked data. Linked data refers to the practice of publishing structured data so that it can be interlinked.

For Wikidata this means that volunteer-contributed data can also be linked to other datasets, databases, and data sources from all around the web and from diverse initiatives outside of the Wikimedia family. For example, Wikidata currently allows interlinking with datasets and databases as diverse as Google Books, Canmore (one of the Historic Environment Scotland databases), the Vatican Library, OmegaWiki, and MusicBrainz.


example of a simple statement consisting of one property-value pair

example of a more complicated statement consisting of one property-value pair, qualifiers, and a reference
By following linked data principles and practices, Wikidata is also able to support and be used by other projects.

Linked data principles
Wikidata uses unique identifiers, or uniform resource identifiers (URIs), for all its items as per linked data standards.

While Wikidata uses a unique data model, its content can be exported in RDF, a widely used and standard format for linked data. In Wikidata terms, a statement is composed of an item and a property-value pair. For those familiar with linked data concepts, an item can be viewed as the subject part of a triplet; the property represents a triplet's predicate; and a value is used to express the object of a triplet.

However, Wikidata statements may also contain elements beyond the subject-predicate-object, such as references and qualifiers (for more information, see Help:Statements). This makes it complicated to fully represent Wikidata's content using the language of RDF.  """
result = chain.invoke({'input': input_text})
print(result)
chain.get_graph().print_ascii() 


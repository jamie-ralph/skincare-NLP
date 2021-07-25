Named entity recognition with skincare routines
================
Jamie Ralph

In this markdown document I will be applying some named entity
recognition (NER) to YouTube skincare routines. In short, NER use a
statistical model to assign a label to an entity (for example: people,
organisations, locations). Here I will be running NER with the spaCy
package in Python with the small version of [their pre-trained English
language pipeline](https://spacy.io/models/en).

## Data pre-processing

I’ll first load up the data and do some pre-processing in R:

``` r
library(here)
library(tidyverse)
library(tidytext)
library(glue)
library(reticulate)

# Get vector of file names
person <- list.files(here("skincare_transcripts")) %>%
    str_remove_all(".txt") %>%
    str_replace_all("_", " ")

# Extract text as character vector
routine_text <- list.files(here("skincare_transcripts"), full.names = T) %>%
    map_chr(read_file) %>%
    str_replace_all("\\r", " ") %>%
    str_replace_all("\\n", " ") %>%
    str_replace_all("- ", "")

# Convert to a tibble for analysis
routine_df <- tibble(
    person = person,
    text = routine_text
)

# Remove sound effects (found inside brackets)
routine_df <- routine_df %>%
  mutate(
    text = str_remove_all(text, "\\s*\\([^\\)]+\\)")
  )
```

## Run NER with spaCy

Now it’s time for the NER. First I’ll import spaCy and import the model,
then print the text data out:

``` python
import spacy

spacy_model = spacy.load("en_core_web_sm")

df = r.routine_df

print(df.head())
```

    ##             person                                               text
    ## 0    alissa ashley  hey guys welcome back to i'm so used to  sayin...
    ## 1  anastasia soare    Well, let's go to bed with me.  Before I go ...
    ## 2  antoni porowski    Hey guys.  Come get ready for bed with me.  ...
    ## 3    ashley graham    Okay, it's time to get unready with me.  Tod...
    ## 4   barbara palvin    Hey.  Come on in.  Let's get ready for bed. ...

Now I’ll use a list comprehension to identify named entities for each
transcript:

``` python
text_list = df.text.tolist()

result = [spacy_model(text) for text in text_list]
```

I’d like to extract the named entities and their labels from the result
objects. I’ll define a function to extract each result as a dataframe,
then concatenate them all together.

``` python
import pandas as pd

def extract_entities(labelled_text):
  
  term = []
  label = []
  
  for word in labelled_text.ents:
    term.append(word.text)
    label.append(word.label_)
    
  return pd.DataFrame({"term": term, "label": label})

skincare_ner = pd.concat([extract_entities(text) for text in result])
```

## Check results

Let’s look at the results of the NER model:

``` python
skincare_ner.value_counts("label", ascending = False)
```

    ## label
    ## CARDINAL       210
    ## PERSON         200
    ## DATE           149
    ## TIME           114
    ## ORG            113
    ## ORDINAL         92
    ## GPE             74
    ## NORP            26
    ## WORK_OF_ART     15
    ## LOC             12
    ## PRODUCT         12
    ## PERCENT          9
    ## FAC              5
    ## QUANTITY         4
    ## EVENT            3
    ## MONEY            2
    ## LAW              2
    ## dtype: int64

The most popular category are CARDINAL, which are numerals not falling
under another type, so we ignore these. Let’s look at the most popular
PERSON tags:

``` python
df = skincare_ner[skincare_ner["label"] == "PERSON"].value_counts("term", ascending = False)

df.head(5)
```

    ## term
    ## Barbara Sturm    8
    ## Sturm            5
    ## Jack Black       4
    ## mm               4
    ## Unicorn Mist     3
    ## dtype: int64

So, we have a few mentions of Barbara Sturm, a luxury skincare brand.
But we also have Unicorn Mist which certainly isn’t a person! This could
be a result of both words being capitalised (I’ll blame Paris Hilton for
that one).

Let’s look at the PRODUCTS category:

``` python
df = skincare_ner[skincare_ner["label"] == "PRODUCT"].value_counts("term", ascending = False)

df.head(5)
```

    ## term
    ## C'est Moi Gentle Makeup Remover Wipes    2
    ## Kleenex                                  2
    ## AHAs                                     1
    ## Benefit Clean                            1
    ## Enzymion                                 1
    ## dtype: int64

These make a little more sense. We have makeup remover wipes, tissues,
chemical exfoliators (AHAs), and a moisturizer.

Finally, let’s look at the ORG label (includes companies, agencies
etc.):

``` python
df = skincare_ner[skincare_ner["label"] == "ORG"].value_counts("term", ascending = False)

df.head(5)
```

    ## term
    ## Biba       5
    ## un         2
    ## CVS        2
    ## Next       2
    ## Sanitas    2
    ## dtype: int64

Some good results here, including skincare brands and CVS Pharmacy.

## Conclusion

NER is an efficient technique for identifying named entities in a large
body of text. The results here weren’t perfect, but the model performed
well considering that it was the small version of spaCy’s models and was
applied to YouTube transcripts, which can be messier than written text.

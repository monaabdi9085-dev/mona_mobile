import pandera.pandas as pa 
import pandas as pd
import sys 

# LLM assistance 
datafil= sys.argv[1] if len (sys.argv) > 1 else "data/kundbas.csv"  # <----------

df = pd.read_csv(datafil)


schema=pa.DataFrameSchema({
    "kund_id":pa.Column(int, unique=True),
    "alder":pa.Column(int, checks=pa.Check.ge(0)),
    "region":pa.Column(str),
    "abonnemang":pa.Column(str),
    "manader_som_kund":pa.Column(int),
    "manadskostnad":pa.Column(float),
    "data_gb_per_manad":pa.Column(float),
    "supportarenden_12m":pa.Column(int),
    "avslutat":pa.Column(int, checks=pa.Check.isin([0,1]))

})

schema.validate(df, lazy=True)

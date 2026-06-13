import pathlib 
import itertools 
p=pathlib.Path(r'c:\Users\hp\anaconda3\Lib\site-packages\statsmodels\tsa\stattools.py') 
text=p.read_text(encoding='utf-8') 
for i,line in enumerate(text.splitlines(),1): 

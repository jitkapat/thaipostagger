# Thai Part-of-speech Tagger in UD style
Proper deployment working in process...

## How to use (for now):

1. clone this repository and copy ALL the contents into your project folder
2. download lastest model file from https://github.com/jitkapat/thaipostagger/releases
3. decompress the contents of the model file into a folder named "bert_base_th" and put in the same folder as tagger.py and util.py

## Example:

```python
from tagger import Tagger
tagger = Tagger()
tagger.tag(yoursentence)
```

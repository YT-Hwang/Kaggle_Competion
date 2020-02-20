# Overview

## What to Predict

- Stage 1 - You should submit predicted probabilities for every possible matchup in the past 5 NCAA® tournaments (seasons 2015-2019).
- Stage 2 - You should submit predicted probabilities for every possible matchup before the 2020 tournament begins.

Refer to the [Timeline page](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/overview/timeline) for specific dates. In both stages, the sample submission will tell you which games to predict.

# Import Packages


```python
import pandas as pd
import numpy as np

import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
#rcParams['figure.figsize'] = 20, 6
%matplotlib inline

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
```

# Data Download


```python
# Downloading the Data using Kaggle API
!kaggle competitions download -c google-cloud-ncaa-march-madness-2020-division-1-mens-tournament
```

    Downloading google-cloud-ncaa-march-madness-2020-division-1-mens-tournament.zip to F:\OneDrive - Georgia State University\Data Science\Competition\Google Cloud & NCAA® ML Competition 2020-NCAAM
    
    

    
      0%|          | 0.00/120M [00:00<?, ?B/s]
      2%|1         | 2.00M/120M [00:00<00:07, 17.3MB/s]
      4%|4         | 5.00M/120M [00:00<00:06, 19.7MB/s]
      7%|6         | 8.00M/120M [00:00<00:05, 21.9MB/s]
      9%|9         | 11.0M/120M [00:00<00:04, 23.7MB/s]
     12%|#1        | 14.0M/120M [00:00<00:04, 25.2MB/s]
     14%|#4        | 17.0M/120M [00:00<00:04, 26.3MB/s]
     17%|#6        | 20.0M/120M [00:00<00:03, 27.1MB/s]
     19%|#9        | 23.0M/120M [00:00<00:03, 27.7MB/s]
     22%|##1       | 26.0M/120M [00:00<00:03, 28.2MB/s]
     24%|##4       | 29.0M/120M [00:01<00:03, 28.5MB/s]
     27%|##6       | 32.0M/120M [00:01<00:03, 23.5MB/s]
     29%|##9       | 35.0M/120M [00:01<00:03, 24.7MB/s]
     34%|###4      | 41.0M/120M [00:01<00:02, 29.0MB/s]
     37%|###7      | 45.0M/120M [00:01<00:02, 29.1MB/s]
     41%|####      | 49.0M/120M [00:01<00:02, 29.0MB/s]
     43%|####3     | 52.0M/120M [00:01<00:02, 29.0MB/s]
     46%|####5     | 55.0M/120M [00:02<00:02, 29.0MB/s]
     48%|####8     | 58.0M/120M [00:02<00:02, 29.1MB/s]
     51%|#####     | 61.0M/120M [00:02<00:02, 29.2MB/s]
     53%|#####3    | 64.0M/120M [00:02<00:03, 18.3MB/s]
     61%|######    | 73.0M/120M [00:02<00:02, 23.8MB/s]
     64%|######4   | 77.0M/120M [00:02<00:01, 25.3MB/s]
     67%|######7   | 81.0M/120M [00:02<00:01, 26.4MB/s]
     71%|#######   | 85.0M/120M [00:03<00:01, 27.2MB/s]
     74%|#######4  | 89.0M/120M [00:03<00:01, 27.8MB/s]
     77%|#######7  | 93.0M/120M [00:03<00:01, 28.2MB/s]
     80%|#######9  | 96.0M/120M [00:03<00:00, 28.6MB/s]
     82%|########2 | 99.0M/120M [00:03<00:00, 28.8MB/s]
     85%|########4 | 102M/120M [00:04<00:01, 15.7MB/s] 
     89%|########9 | 107M/120M [00:04<00:00, 19.8MB/s]
     95%|#########4| 114M/120M [00:04<00:00, 24.8MB/s]
     98%|#########8| 118M/120M [00:04<00:00, 26.0MB/s]
    100%|##########| 120M/120M [00:04<00:00, 28.2MB/s]
    

# Data Import


```python
# Getting all the files in the directory.
def existing_file_list(path):
    ''' Extracting File Names '''

    allFiles = glob.glob(path + "/*.csv")
    new_list = []
    for i in allFiles:
        before = 'original\\'
        after = '_minute'
        ticker = i[i.find(before) + len(before) : i.find(after)]
        new_list.append(ticker)    
    #list_ticker = list(filter(None, text))  # drop all the empty elements and put them in a list
        
    return (new_list)
```


```python
datapath = 'F:\\OneDrive - Georgia State University\\Data Science\\Competition\\Google Cloud & NCAA® ML Competition 2020-NCAAM\\Data\\raw\\'

from os import listdir
from os.path import isfile, join
main_data_list = [f for f in listdir(datapath) if isfile(join(datapath, f))]

datapath
main_data_list
```




    'F:\\OneDrive - Georgia State University\\Data Science\\Competition\\Google Cloud & NCAA® ML Competition 2020-NCAAM\\Data\\raw\\'






    ['MEvents2015.csv',
     'MEvents2016.csv',
     'MEvents2017.csv',
     'MEvents2018.csv',
     'MEvents2019.csv',
     'MPlayers.csv',
     'MSampleSubmissionStage1_2020.csv']




```python
#element_types = pd.read_csv(datapath + 'element_types.xlsx',index_col='id', engine = 'python')
bootstrap = pd.read_excel(datapath + 'bootstrap.xlsx',index_col='id')
main_data = pd.read_csv(datapath + 'merged_gw.csv', index_col=['id', 'fixture'], engine = 'python')
MPlayers = pd.read_csv(datapath + 'MPlayers.csv', index_col='PlayerID, TeamID')

```

# Data Exploratory Analysis


```python

```

# Reference

- Primary: [google-cloud-ncaa-march-madness-2020-division-1-mens-tournament](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament)
- Secondary:

# Description

The task is to predict the start and end of human steps from a set of 6 signals. The data was recorded using accelerometer and gyroscope sensors, which were held in the hand during 4 different activities, including simple walking, walking up and down stairs, and standing still.

#### Training data

The training data consists of folders, each containing the corresponding files:

- a `xxxx.csv` file with seven columns (one for activity id and three (x,y,z) each for accelerometer and gyroscope data).
- a `xxxx.csv.stepMixed` file with step labels (containing start and end indexes for each complete step).

  Sometimes, there are more of these files within the folder. These are chunks of the measured signals just to make the files smaller. You shall use all of these data.

#### Testing data

For testing your models, you are provided with a `testData.csv` file that has the same structure as the to the training inputs `xxxx.csv`, but without the activity column.

Your goal is to produce an output file, which gives for each times point in the signal a probability of a start and end of a step. E.g. \`15, 0.7, 0.5\` means: the probability that time point 15 is the beginning of a step is 0.7 and the probability that time point 15 is the end of a step is 0.5.

The expected format for the output file is as follows:

```
index,start,end
0,0.01,0.2
1,0.9,0.1
2,1.0,0.0
3,0.05,0.95
.
.
.
102090,0.0,1.0
```
## 

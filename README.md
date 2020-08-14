# Issue Complexity Prediction by Issue Content
**OR:** How the GitHub "good first issue" functionality works / could be improved upon.

Technologies Used:
- [Jupyter Notebooks]() (Statistical Testing & Reporting)
- [Beautiful Soup]() (GH Trending Page Scraping)
- [PyGitHub]() (GitHub API Access / Corpus Building)
- [PyYAML]() (Corpus Persistence)
- [Keras CPC]() (Complex Algorithm Approach)
- div. other python packages assoc. w/ ML

# Getting Started
## Setup
1. Make sure you have `python` and `docker` installed.
2. Clone the repository.
3. Install the requirements via `$ pip install -r requirements.txt`
4. Get your Personal GitHub API Access Token [here]() 1. Put the AccessToken into a ".env" file inside the root repo folder

   as follows:

   ```shell script
   GITHUB_ACCESS_TOKEN=<your_access_token>
   ```

5. If you want to use a prepared GPU ready tensorflow setup run:

   ```shell script
   $ docker build -t hv10/crmproject tfcontainer/
   ```

   inside the repository folder.

   This only makes sense if you want to train the Neural Networks yourself
   and if you are unsure about how to setup tensorflow+gpu correctly.
1. To run most of the scripts you will need to install this repository as a dev. package in python.
   - Don't worry all that this means is that python is going to symlink this repo as a package 
   instead of actually copying the contents into the installation.
   - Run the following inside the repository:
        ```shell script
        $ pip install -e .
        ```
1. You should be ready now :)
   
   

## Usage

### Included Artifacts

To make the work more reproducible some artifacts are included in the repository.
They represent information used to archieve certain statistical test and can be reused to (hopefully) arrive at similar 
conclusions.

The artifacts included are:
- a list of all repos considered for crawling (`trending_repos.repo`)
- a list of the issues actually crawled (`corpus/collected_issues.csv`)
    - the actual issue are not included out of copyright concerns
    - as well as size concerns (~56GB of raw data)
- a list of the subselection of 1000 issues representing
    - the "good first issues" used for statistical tests (`experiments/notebooks/df_gfi_1000.csv`)
    - the "normal issues" used for statistical tests (`experiments/notebooks/df_ngfi_1000.csv`)
- the vectorizers trained on the subsets of the whole corpus (which are used in the Keras Models)
    - `models/vectorizer_model_<200/2000/6000/20000>` -- trained on 200 / 2000 / 6000 / 20000 randomly selected issues
    - for the later experiments only the `20000` model was used
- a fully trained DNN & CNN (<!--TODO-->)
- a fully pre-trained CPC model (<!--TODO-->)
- a fully downstream-trained CPC model (<!--TODO-->)

### Building the Corpus

To build the corpus you have multiple options.

The first is to just download issues from GitHub.
For that you can use the `github_trending_scraper.py`. 
The script will collect the currently trending english speaking repositories.
The script will output the collected repositories into a file located inside the parent directory of the script.

The `.repo` file can then be used as input for the `global_corpus.py`.
That script will start building the corpus inside the current working directory.

Structurally it will look like this:
```shell script
corpus
    |-<owner>
      |-<repo_name>
        |- issueXXXX.yaml
        |- issueXYXY.yaml
        ...
      ...
    |-<owner2>
      |-<repo_name>
      ...
``` 

It will try to collect **all** issues from the named repositories.

Using the `good_first_issues.py` you can collect issues with the aforementioned label.
This will basically just search via the GitHub API for repositories with at least five issues with the label.

**Note:** The script will only download the issues with the label.

<!--TODO--> The other option is to download the corpus via the `corpus_from_csv.py`.

### Making the Vectorizer

The vectorizer can be build by calling the `experiments/algorithms/vectorization.py` script.
It expects the first CLI argument to be the path to the corpus, 
the second one will be the sample size used for training.

**Note:** Be mindful about the amount of data you really want to train with it.
It takes a while, and it doesnt show the progress... Which is a tensorflow issue. 

On my machine (8th-gen i5) it took about 6h for 20.000 samples.

### The statistical Tests

The statistical tests are fully self-contained Jupyter-Notebooks, which can be run via jupyter.

The data used in the statistical tests is also located in the same folder to enable a smooth experience.

### Training the ML Models

### Training & Finetuning CPC 
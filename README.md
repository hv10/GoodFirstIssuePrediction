---
description: >-
  OR: How the GitHub "good first issue" functionality works / could be improved
  upon.
---

# Issue Complexity Prediction by Issue Content

Technologies Used:

* [Jupyter Notebooks](./) \(Statistical Testing & Reporting\)
* [Beautiful Soup](./) \(GH Trending Page Scraping\)
* [PyGitHub](./) \(GitHub API Access / Corpus Building\)
* [PyYAML](./) \(Corpus Persistence\)
* [Keras CPC](./) \(Complex Algorithm Approach\)
* div. other python packages assoc. w/ ML

## Getting Started

### Setup

1. Make sure you have `python` and `docker` installed.
2. Clone the repository.
3. Install the requirements via `$ pip install -r requirements.txt`
4. Get your Personal GitHub API Access Token [here](./) 1. Put the AccessToken into a ".env" file inside the root repo folder

   as follows:

   \`\`\`shell script

   GITHUB\_ACCESS\_TOKEN=

   \`\`\`

5. If you want to use a prepared GPU ready tensorflow setup run:

   \`\`\`shell script

   $ docker build -t hv10/crmproject tfcontainer/

   \`\`\`

   inside the repository folder.

   This only makes sense if you want to train the Neural Networks yourself

   and if you are unsure about how to setup tf-gpu correctly.

6. To run most of the scripts you will need to install this repository as a dev. package in python.
   * Don't worry all that this means is that python is going to symlink this repo as a package

     instead of actually copying the contents into the installation.

   * Run the following inside the repository:

     \`\`\`shell script

     $ pip install -e .

     \`\`\`
7. You should be ready now :\)

### Basic Usage

#### Included Artifacts

To make the work more reproducible some artifacts are included in the repository. They represent information used to archieve certain statistical test and can be reused to \(hopefully\) arrive at similar conclusions.

The artifacts included are:

* a list of all repos considered for crawling \(`trending_repos.repo`\)
* a list of the issues actually crawled \(`corpus/collected_issues.csv`\)
  * the actual issue are not included out of copyright concerns
  * as well as size concerns \(**~56GB** of raw data\)
* a list of the subselection of 1000 issues representing
  * the "good first issues" used for statistical tests \(`experiments/notebooks/df_gfi_1000.csv`\)
  * the "normal issues" used for statistical tests \(`experiments/notebooks/df_ngfi_1000.csv`\)
* the vectorizers trained on the subsets of the whole corpus \(which are used in the Keras Models\)
  * `models/vectorizer_model_<200/2000/6000/20000>` -- trained on 200 / 2000 / 6000 / 20000 randomly selected issues
  * for the later experiments only the `20000` model was used
* a fully trained DNN & CNN \(\)
* a fully pre-trained CPC model \(\)
* a fully downstream-trained CPC model \(\)

#### Building the Corpus

To build the corpus you have multiple options. The first is to active


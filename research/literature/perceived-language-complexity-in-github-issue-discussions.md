---
description: provides a tool to calculate project-specific perceived language complexity
---

# Perceived language complexity in GitHub issue discussions

### Links

* PDF: [here](https://web.cs.ucdavis.edu/~filkov/papers/language_issues.pdf)
* IEEE-entry: [here](https://ieeexplore.ieee.org/abstract/document/8115620)

## Summary

### Introduction

* users have different social and cultural identities
  * socioeconomic status, cultural bg, etc.
  * influences how they talk, listen, understand
* GitHub and OSS projects take on the role of communities in sociological sense
* have specific language \(project-specific terms, related jargon\)
* using correct language as mark of standing
* issues combine narrative style w/ technical arguments
* study language complexity and perceived complexity
* ~90ths issues, 460ths posts

#### Speech Communities & Communities of Practice

> A speech community \(SC\) describes a group of people who use language in a way that is mutually accepted among the group

> \[Here\] an example where a member of the out-group \(an end-user\) has an issue. They attempt to express their problem,but the contributing member \(in-group\) is unable to understand the description, and further asks the end-user to conform to project-specific norms \(i.e., fill out the standard issue report form\)

> members of a CoP __\[Community of Practice\] do not have to be physically co-located,but can form a “virtual community of practice” with the same attributes as a standard CoP

> Llamas et al. present an example of a CoP within the workplace, saying that individuals regularly engage in scheduled social practices \(e.g.,business meetings\), and mutually define themselves as CoP members

#### Entropy and Language Complexity

> There has been work in using entropy as a description of language complexity and style.

> Repetitive language style, as in the Bible, has lower entropy and is easier to understand and read, while the style of, e.g., James Joyce seems more complex and harder to read.

> Perceived language complexity is the distance between a community’s language and a given text. This is quantified using the \(cross-\) entropy distance.

### Research Questions

* Is  there  evidence for a standard for the GitHub community language? Does this also carry over to projects,i.e., is there evidence for a project-specific  language?
* Is there migration in perceived language complexity, over time, toward the project norm?
* Do users \(popular or experienced\) conform to their associated project language? And does perceived language complexity influence popularity?
* What is the relationship, if any, between perceived language complexity and issue resolution latency?

### Data & Methods

> Our data is  a sample of 48 projects from the top 900 most starred and followed projects.

> The number of stars and followers are proxies for project popularity, and can identify projects likely to  contain enough issues to build robust language models.

> For every post in an issue thread,  we  gathered  the  following  information:  date  of  post,post body, login of poster, and issue closing time. Post bodies were  used  to  extract  text-related  metadata  and  to  build  our language models

> In addition to examining comments we also collected  a  variety  of  metadata  from  projects.  This  included commit-related  metrics:  number  of  lines  added  and  deleted, date of commit, commit author; and user data: full name, time they joined GitHub, and location.

> We  constructed  a  social  network  for  each  project  using @mentions in  their  issue  comment  threads.

* they also tracked the network over time
  * key metric: _in-degree_ 
  * _out-degree_, _between-ness_ and _degree centrality_ were tracked but not used
* for each project a _global_ and a _local_ corpus
  * global: corpus from all projects except current
  * local: just the current project
* use _n-gram_ model instead of _lstm_ bc. of computational costs


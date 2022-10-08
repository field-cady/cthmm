# cthmm

This is a python library for training and applying
[continuous-time markov hidden markov models](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain) (CT-HMMs).
They are a simple, but exceptionally powerful tool for extracting
signal from a stream of noisy data.

It was developed by [Field Cady](http://www.fieldcady.com).


# Basic Usage

A Jupyter Notebook with usage usage is [here](https://github.com/field-cady/cthmm/blob/main/CTHMM%20Examples.ipynb).
The main functionalities are to:
* Create a CT-HMM with pre-defined parameters
* Fit a CT-HMM to data (i.e. one or more sequences of observations, along with their timestamps)
* Given a sequence of observations and timestamps, use a fitted CT-HMM to decode the underlying states
    at each timestamp

# TODO Items
* Support for GaussianCTHMM and ExponentialCTHMM by inheriting from BaseCTHMM class
* Allowing human-readable names for states
* Performance improvements, possibly by converting to C or Cython
* Unit testing
* Add in some fitting tolerances to reduce overfitting


# What is a continuous-time hidden markov model?

Say you take somebody's temperate and it's in the normal range.
In isolation that suggests they are healthy rather than sick.
But if they had elevated temperatures one hour before and after,
chances are that measurement was a fluke and they were actually sick the whole time.

Hidden markov models address this problem more generally.
There is some underlying state S of the world that varies over time
(or something like time - things just have to be sequential),
and you have some idea of how often S changes.
You don't know S for certain, but you have a sequence of
imperfect observations O at certain points in time.
You can use O(t) to guess at S(t), but you want to use the
other observations before/after t to improve on that guess.

In a traditional HMM time occurs in discrete steps:
t is an integer, S(t) is either the same or different from S(t-1),
and you have the accompanying observation O(t).
But in many real-world situations time flows continuously,
and S can change at any moment.
Observations come at irregular intervals,
and they may or may not be close to the times that S changes.
If you are trying to guess S(t), the surrounding observations
are more/less informative depending on how far they are from t.

In math terms a traditional HMM assumes that S is a
discrete-time [Markov Chain](https://en.wikipedia.org/wiki/Markov_chain),
while a CT-HMM assumes that S is a
[continuous-time markov model](https://en.wikipedia.org/wiki/Continuous-time_Markov_chain)
and makes no assumption about what observations get taken.
The nuts-and-bolts math of a CT-HMM ends up being a lot trickier
than for a normal HMM, and partly for that reason they are not
used as commonly.
But in many situations they are essential.


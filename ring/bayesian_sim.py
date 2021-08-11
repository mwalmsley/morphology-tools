"""

read "ring" as "expert says it's a ring"



Loss function for network, p observed data given variable of interest data
p(mobile|f(x)) / N =  p(ring) log p^hat(ring) + p(~ring) log p^hat(~ring)

Normally:
p^hat_ring = f(x), use softmax to ensure [0, 1] interval
p(ring) = 1 if labelled ring, 0 otherwise

Here, uncertain labels:
p(ring) = p(ring|mobile votes), need to estimate. Nice separable problem (and useful in of itself).


p(ring|mobile votes) = p(mobile votes|ring) p(ring) / [p(mobile votes|ring)p(ring) + p(mobile votes|~ring)p(~ring)]

p(ring), p(~ring) are the base rates of expert labels for gz mobile galaxies
p(mobile|ring) is the histogram of mobile responses for expert-labelled rings

But these all implicity assume they were voted on by gz mobile. Explicitly:

p(ring|mobile votes, voted) = p(mobile votes|ring, voted) p(ring, voted) / [p(mobile votes|ring, voted)p(ring, voted) + p(mobile votes|~ring, voted)p(~ring, voted)]

Most galaxies would not be selected, so classifier trained to estimate p(ring|mobile votes, voted) would be miscalibrated

Need a correction to reduce estimate according to the rate of expert rings

Bayes rule to calibrate vs. base rates
p(ring|f(x)) = p(f(x)|ring)p(ring)/ [ p(f(x)|ring)p(ring) + p(f(x)|p(~ring)) ]


Now p(ring) and p(f(x)|p(ring)) are the histograms/rates for expert-labelled rings in random (from the selected subset) galaxies
Potentially tedious to estimate - may need to find say 100 rings -> few thousand random galaxies, or 30 rings -> 1000 galaxies. 
"""
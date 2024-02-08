# Nested Shapley

The following repository contains functions to compute the true Shapley value and the Nested Shapley as explained in Alonso, R., Pisciella, P., Crespo del Granado, P. (2024) Fair Investment Strategies in Large Energy Communities: A Scalable Shapley Value Approach (under review).

The different functions for computing the allocation methods are included folder `src`.

Several examples of applications are presented in the folder `examples`. Furthermore, the data for the demand and prices
included in `data_paper`. The prices for 2019 were retrieved from [ENTSOE Transparency platform](https://transparency.entsoe.eu/dashboard/show) using their API.
The loads were generated using the [LoadProfileGenerator](https://www.loadprofilegenerator.de/) software developed by Pflugradt et al. (2022).

**References**

Pflugradt et al., (2022). LoadProfileGenerator: An Agent-Based Behavior Simulation for Generating Residential Load Profiles. Journal of Open Source Software, 7(71), 3574, https://doi.org/10.21105/joss.03574

# Nested Shapley

The following repository contains functions to compute the true Shapley value and the Nested Shapley as explained in Alonso, R., Pisciella, P., Crespo del Granado, P. (2024) Fair Investment Strategies in Large Energy Communities: A Scalable Shapley Value Approach (under review).

The different functions for computing the allocation methods are included in folder `src`.

Also, the stratified expected value from Cremers et al. (2023) is computed in the Jupyter notebook `examples/appendix_stratifiedexpected.ipynb`for comparison with the true Shapley and the Nested Shapley value.

Several examples of applications are presented in the folder `examples`. Furthermore, the data for the demand and prices
are included in `data_paper`. The prices for 2019 were retrieved from [ENTSOE Transparency platform](https://transparency.entsoe.eu/dashboard/show) using their API.
The loads were generated using the [LoadProfileGenerator](https://www.loadprofilegenerator.de/) software developed by Pflugradt et al. (2022).
The solar profiles were obtained through [Renewables Ninja](https://www.renewables.ninja/) developed by Pfeninger et al.(2016) and Staffell et al. (2016)

**References**

Cremers, S., Robu, V., Zhang, P., Andoni, M., Norbu, S., & Flynn, D. (2023). Efficient methods for approximating the Shapley value for asset sharing in energy communities. Applied Energy, 331, 120328. https://doi.org/https://doi.org/10.1016/j.apenergy.2022.120328 

Pflugradt et al., (2022). LoadProfileGenerator: An Agent-Based Behavior Simulation for Generating Residential Load Profiles. Journal of Open Source Software, 7(71), 3574, https://doi.org/10.21105/joss.03574

Pfenninger, Stefan and Staffell, Iain (2016). Long-term patterns of European PV output using 30 years of validated hourly reanalysis and satellite data. Energy 114, pp. 1251-1265. doi: 10.1016/j.energy.2016.08.060

Staffell, Iain and Pfenninger, Stefan (2016). Using Bias-Corrected Reanalysis to Simulate Current and Future Wind Power Output. Energy 114, pp. 1224-1239. doi: 10.1016/j.energy.2016.08.068

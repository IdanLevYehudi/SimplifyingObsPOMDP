# Simplifying Complex Observation Models in Continuous POMDP Planning with Probabilistic Guarantees and Practice #

This is the python code for the following paper to be presented at AAAI-24, in which we analyze and provide probabilistic guarantees for simplifying observation models in POMDP planning.

# Citation #

We kindly ask to cite our paper if you find this code useful.

- Lev-Yehudi, I.; Barenboim, M.; and Indelman, V. 2024.
Simplifying Complex Observation Models in Continuous
POMDP Planning with Probabilistic Guarantees and Prac-
tice. In AAAI Conf. on Artiﬁcial Intelligence.

```
@inproceedings{LevYehudi24aaai,
  author =   {I. Lev-Yehudi and M. Barenboim and V. Indelman},
  booktitle =  AAAI,
  location =   {Vancouver, Canada},
  month =  {February},
  title =  {Simplifying Complex Observation Models in Continuous POMDP Planning with Probabilistic Guarantees and Practice},
  year =   2024,
}
```

# File Description #

- environment.yml - environment file for conda.
- src/beacons_pomdp_test.py - Main script for running scenarios with either original or simplified models.
- src/beacons_test.py - Used to test only the environment.
- src/mes_simp_test.py - Used to test the measurement simplification, calculation of Delta_Z and m_i.
- src/bounds_analyzer.py - Used to generate figures in paper.
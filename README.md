# ChemCluster
Easy-to-use python module for clustering a database of 4,591 unique optically active organic molecules according to user-defined similarity queries.

`chemcluster` combines molecular fragmentation and chemical intuition to identify the core groups, bridges and substituents of a database of molecules in the domain of organic spectroscopy. These classified fragments are then used to create a novel fingerprint for each molecule that allows more chemically relevant similarity comparisons than afforded using traditional fingerprinting schemes, such as Extended-Connectivity FingePrints (ECFPs). The novel similarity comparisons are used to generate clusters of similar molecules which can be drawn together on file along with their computational data so that trends in the absorption properties can be studied. The large repository of classified fragments generated as part of this application may also be used for fragment-based materials discovery.

Additionally, `chemcluster` exposes a number of functions for obtaining an overview of the database in terms of the distributions of key descriptors and prominent groups. It also provides an interface for accessing a wide range of properties of the molecules and fragments, including access to their RDKit.Mol representations.

This code was developed for the Molecular Engineering group at University of Cambridge as part of my MPhil in Scientific Computing. The database of molecules and their computational properties was created by the Molecular Engineering group.

## Functionality

* Apply BRICS fragmentation to ~4,500 optically active organic molecules
* Classify fragments as: core group, substituent, or bridge
* Generate a novel fingerprint for each molecule using its classified fragments
* Cluster the database into subsets of molecules that share some or all of their novel fingerprints (with or without counts)
* Generate a similarity report for any two molecules using common fingerprints (ECFP, FCFP, RDKit and MACCS) and metrics (Tanimoto and Dice)
* Provide an interface for exploring the database according to user-defined SMARTS patterns
* Generate distribution plots of key descriptors and computational data for the molecules in the database
* Calculate the average similarity among a group of molecules using a range of common fingerprints
* Cluster core groups using Butina clustering according to their similarity using a range of common fingerprints

## Using the module
It is highly recommended that the code snippets below are run from an interactive Python console such as IPython. In this way, images of molecules are printed to screen and exploratory analysis is simplified greatly. It will be assumed that these snippets are run interactively.
### Run the Fragmentation Algorithm and Analyse Fragments

Firstly it is necessary to generate and classify the fragment entities. The first time this is run it will take a couple of minutes. The objects will then be saved to pickle files and should be loaded instantly in future.

As an overview, the 40 most frequently occurring entities of each class (raw BRICS fragments, core groups, substituents and bridges) are drawn to disk. The directory where images are drawn to can be altered in `config.json`. The 'frequency' refers to the number of molecules in which the fragment appears at least once.

```python
from chemcluster import ChemCluster

cc = ChemCluster()
cc.generate_fragments() #takes a few minutes

cc.draw_fragments(from_idx=0, to_idx=40)
cc.draw_groups(from_idx=0, to_idx=40)
cc.draw_substituents(from_idx=0, to_idx=40)
cc.draw_bridges(from_idx=0, to_idx=40)
```
The unique ID of each entity is drawn as part of the legend so that the corresponding entity object can be accessed and its properties explored:

```python
group = cc.get_group(group_id=209)
group #draw to interactive console
group.get_size() #number of heavy atoms
rdk_mol = group.get_rdk_mol() #RDKit.Mol object

#Draw the molecules that contain this group
cc.draw_entity_mols(ent_type='group', ent_id=209)
```
where the options for `ent_name` are: group, fragment, substituent and bridge.

### Generate the Novel Fingerprints
Using the classified fragments we can assign a novel fingerprint to each molecule:

```python
cc.generate_novel_fps()

#choose a sample molecule and draw to console
mol = cc.get_molecule(mol_id=3321)
mol #wavelength and strength in legend

#output fingerprint to console
mol.get_novel_fp()
```
This will output both a text representation of the fingerprint and a drawing of each fragment entity (no duplicates). The format of the text representation is:
```
bridges: (bridge_id, count), ...
groups: (group_id, count), ...
subs: (sub_id, [(group_id, group_instance), ...]), ...
```

### Generating Molecule Clusters Using Queries
Using the novel fingerprinting scheme, molecules can be clustered into specific subsets using queries. We will take a look at 4 such queries:
* molecules with the same core groups (with counts)
* molecules with the same atoms and bonds but different topology/connectivity
* molecules with the same fingerprint except for the counts of one of the groups
* molecules that only differ by a (user-specified) substituent

Molecules with the same core groups (with counts) only differ by their substituents and their bridges, which may reveal some interesting substituent/bridge effects.
```python
from chemcluster import ChemCluster

cc = ChemCluster()

cc.generate_fragments()
cc.generate_novel_fps()

cc.q_same_groups(counts=True, draw=True)
```
This will produce a lot of files, one per subset (or multiple per subset if there are more than 20 molecules in the subset). The files are named using the group fingerprint i.e. the groups the molecules in that file share in common. For example, a file named '(2,2)\_(8,1).1-20.png' means the molecules in this file all have 2 instances of group 2, and 1 instance of group 8 (and that its the first 20 molecules of this subset, hence 1-20).

Since there are many files, it would be useful to only draw those subsets where there is a large computational variance between the molecules in the subset. This can be achieved by setting `large_comp_diff=True`.

If the user is only interested in subsets that contain a particular group, for example azo, then they can specify the SMARTS pattern of this group in patterns.txt and modify the query: `cc.q_same_groups(counts=True, pattern='azo', draw=True)`.

Finally, it might be of benefit to retrieve the molecule objects in a particular subset. For example, to calculate the average similarity between them using an existing fingerprinting scheme. The is the purpose of the `query` parameter, which needs to be set to the name of the file from which the subset originates (only the fingerprint segment of the file).
```python
mols = cc.q_same_groups(counts=True, query='(2, 2)_(8, 1)', draw=False)
cc.average_cluster_sim(mols=mols, fp_type='morgan', metric='dice')
```

Molecules that differ only by topology can be clustered using:
```python
cc.q_diff_topology(draw=True, large_comp_diff=True)
```

In order to study molecules that differ by the counts of one particular group, we must select a group to isolate. A naturally interesting choice is the thiophene group since it is used as a building block unit in pi-bridges.

We need to provide the group_id of the group we wish to isolate, so the first step is specify the thiophene SMARTS in patterns.txt and draw to disk all groups that contain this pattern, to determine the ID of the group we want. In this case we just want the basic thiophene unit which has an ID of 31.

```python
#check images to find the correct ID
cc.get_groups_with_pattern(pattern='thiophene', draw=True)
cc.q_diff_counts(group_id=31, draw=True)
```

For the final query, we will choose to analyse those molecules where the only difference is the presence or absence of a cyanide group.

```python
#check images to find the correct ID
cc.get_subs_with_pattern(pattern='cyanide', draw=True)
cc.q_diff_substituents(sub_id=114, draw=True)
```

### Plotting Database distributions

A number of convenient plots and distributions are defined in the `metrics.py` file. Only some of these are exposed via the `ChemCluster` class, but the `Metric` classes can of course be imported and used, provided they are instantiated with the necessary `Data` objects (as in `chemcluster.py`). Three example distribution plots are shown below.

```python
from chemcluster import ChemCluster

cc = ChemCluster()

cc.novel_fp_scatter(min_size=15, counts=True)
cc.comp_dist_plot()
#SMARTS for these groups must be in patterns.txt
groups = ["coumarin","azo","anthraquinone",
          "triarylmethane","thiophene","benzothiazole"]
cc.group_violin_plots(groups)
```
### Convenience Functions
We can get a similarity report for any two molecules:
```python
cc.similarity_report(mol_ids=[1500, 1510])
```
| Fingerprint      | Metric | Similarity
| ----------- | ----------- | ----------
|MACCS        |dice      |0.475248
|      |tanimoto  |0.311688
|Morgan       |dice      |0.26
|     |tanimoto  |0.149425
|RDKit          |dice      |0.632669
|       |tanimoto  |0.462703
|Morgan feature  |dice      |0.268293
|  |tanimoto  |0.15493
|Topology     |dice      |0.4
|     |tanimoto  |0.25

We can also find the average similarity across the entire database of a certain fingerprint/metric combination. This also returns an estimate for the average distance from the true similarity value in terms of wavelength similarities.
```python
avg_fp, avg_error = cc.average_fp_sim(fp_type='rdk',
                                      metric='dice')
```

The core groups can be clustered using the Butina clustering algorithm and a fingerprint/similarity of choice. I have also developed my own basic clustering algorithm but this is still in progress.
```python
cc.generate_fragments()
cc.draw_group_clusters(cutoff=0.2, fp_type='MACCS', similarity='dice')
```
## Installation & Requirements

To install, simply clone the repository:
```
git clone https://github.com/PjFlan/chemcluster.git
```
and open the directory using an interactive development environment such as Spyder. It is highly recommended to install Anaconda to manage packages and run this code in Spyder. See (https://docs.anaconda.com/anaconda/user-guide/getting-started/).

The module requires the following packages (all can be installed via conda):

* pandas (https://anaconda.org/anaconda/pandas)
* rdkit (https://www.rdkit.org/docs/Install.html)
* pymongo (https://pymongo.readthedocs.io/en/stable/)
* seaborn (https://seaborn.pydata.org/)
* tabulate (https://pypi.org/project/tabulate/)
* matplotlib (https://matplotlib.org/3.1.1/users/installing.html)

rdkit may be installed via conda (recommended):
```
conda install -c rdkit rdkit
```

The MongoDB version of the database (recommended) can be downloaded from (https://doi.org/10.6084/m9.figshare.7619672.v2) although it is not strictly necessary, since the relevant data has been dumped to text files inside the /input directory. There is a flag in the `config.json` file called 'db' for reloading data from the database but this is turned off by default.

## Configuration

The file `config.json` contains a number of configuration settings. The options should not be deleted, though the values can be changed. The main parameter that might need to be changed is the `root` directory, which specifies the base directory for all output files (images and plots).

The `tmp` parameter can be switched on if just looking to quickly inspect a cluster of molecules. The files in this folder are deleted the next time tmp is used.

The `regenerate` option can be used to rerun any of the algorithms from scratch i.e ignore and pickle files or .txt files.

Finally, there is logging functionality implemented in each class. This is a wrapper over the Python logging module, where each class has its own child logger so that the name of the class from which the message was logged is output to the log message. Every class logger has a level of DEBUG which should not be changed. Instead, to change the verbosity, adjust the logging level in `config.json` to one of 10, 20, 30, 40 or 50.

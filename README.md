# chemcluster
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

Firstly it necessary to generate and classify the fragment entities. Then we will draw the 40 most frequently occurring of each class (raw BRICS fragments, core groups, substituents and bridges) to disk. Here the frequency refers to the number of molecules in which the fragment appears at least once.
```python
from chemcluster import ChemCluster

cc = ChemCluster()
cc.generate_fragments()

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
cc.draw_entity_mols(ent_name='group', ent_id=209)
```
where the options for `ent_name` are: group, fragment, substituent and bridge.

### Generate the Novel Fingerprints
Using the classified fragments we can assign a novel fingerprint to each molecule:

```python
cc.generate_novel_fps()

#choose a sample molecule and draw to console
mol = cc.get_molecule(mol_id=3321)
mol

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
This will produce a lot of files, one per subset (or multiple per subset if there are more than 20 molecules in the subset). The files are named using the group fingerprint i.e. the groups the molecules in that file share in common. For example, a file named '(2,2)\_(9,1).1-20.png' means the molecules in this file all have 2 instances of group 2, and 1 instance of group 9 (and that its the first 20 molecules of this subset, hence 1-20).

Since there are many files, it would be useful to only draw those subsets where there is a large computational variance between the molecules in the subset. This can be achieved by setting `large_comp_diff=True`.

If the user is only interested in subsets that contain a particular group, for example azo, then they can specify the SMARTS pattern of this group in patterns.txt and modify the query: `cc.q_same_groups(counts=True, pattern='azo', draw=True)`.

Finally, it might be of benefit to retrieve the molecule objects in a particular subset. For example, to calculate the average similarity between them using an existing finerprinting scheme. The is the purpose of the `query` parameter, which needs to be set to the name of the file from which the subset originates (only the fingerprint segment of the file).
```python
mols = cc.q_same_groups(counts=True, query='(2,2)\_(9,1)', draw=False)
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

cc.q_diff_substituents(sub_id=114, draw=True,
                      large_comp_diff=True)
```

### Plotting Database distributions

A number of convenient plots and distributions are defined in the `metrics.py` file. Only some of these are exposed via the `ChemCluster` class, but the `Metric` classes can of course be imported and used, provided they are instantiated with the necessary `Data` objects (as in `chemcluster.py`).

```
from chemcluster import ChemCluster

cc = ChemCluster()
cc.novel_fp_scatter(min_size=15, counts=True)
cc.comp_dist_plot()

#SMARTS for these groups must be in patterns.txt
groups = ["coumarin","azo","anthraquinone",
          "triarylmethane","thiophene","benzothiazole"]
cc.group_violin_plots(groups)
```
## Installation & Requirements

To install, simply clone the repository:
```
git clone https://github.com/LiamWilbraham/pychemlp.git
```
and add the location of the pychemlp repository to your PYTHONPATH.

The module requires the following packages (all can be installed via conda):

* tensorflow (https://www.tensorflow.org/install/)
* scikit-learn (https://scikit-learn.org/stable/install.html)
* pandas (https://anaconda.org/anaconda/pandas)
* numpy (https://anaconda.org/anaconda/numpy)
* rdkit (https://www.rdkit.org/docs/Install.html)

rdkit may be installed via conda (recommended):
```
conda install -c rdkit rdkit
```

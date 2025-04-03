# Composite Alluvial Plot

![Composite Alluvial](Composite_Alluvial.png)

This Python script creates a composite alluvial plot of the flow of units across different sequential categories. 

## How to run
First, make sure you have all the dependencies! run in the Command Prompt or Terminal
```
 pip install -r requirements.txt
```
To use, run in the Command Prompt or Terminal
```cmd
python .\Composite_alluvial.py <Pandas dataframe CSV filepath>
```
where `<Pandas dataframe CSV filepath>` is a Comma-Separated Variables file directory string where the first column are unit ids and the other columns are in a time series of unit categorization or membership. For example:
```
id,T1,T2,T3
0,1,1,2
1,2,1,2
2,1,2,2
3,2,2,1
```
where each row represents a single unit's membership in categories (1, and 2) in timeframes (T1, T2, and T3), and the first column contains unique identifiers.

Alternatively, you can run the `plot_df(df)` function in `Composite_alluvial.py` where `df` is a Pandas dataframe as described above.

To see an example visualization, you can just run 
```cmd
python .\Composite_alluvial.py
```
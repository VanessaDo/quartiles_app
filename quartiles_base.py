# Quartiles App

import numpy as np
import pandas as pd
from bokeh.models import HoverTool, ColumnDataSource, FactorRange
from bokeh.plotting import ColumnDataSource, figure, output_file, show

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_rows', None)

def readData(file_name):
    """Reads data in csv format into a pandas data frame"""
    df = pd.read_csv(file_name)
    return df

def sortIndex(df):
    """Takes a dataframe, sorts the data ascending by the 'VALUE' column and
    assigns a row order reference"""
    df_sorted = df.sort_values(by=['VALUE'])
    df_sorted['ROW_REF'] = np.arange(1,len(df_sorted)+1)
    return df_sorted

def allocateIdealQuartileCount(df):
    """Calculates the number of data points to include in each quartile
    bucket"""
    num_rows = len(df)
    q_idealcountarray = np.repeat(num_rows // 4, 4)
    q_remainder = num_rows % 4
    if q_remainder == 1:
        q_idealcountarray[0] += 1
    elif q_remainder == 2:
        q_idealcountarray[[0, 2]] += 1
    elif q_remainder == 3:
        q_idealcountarray[[0,1,2]] += 1
    return q_idealcountarray

def getQuartileThresholdIndex(q_idealcountarray):
    """Gets the index of the last item in each quartile"""
    i1 = q_idealcountarray[0]
    i2 = sum(q_idealcountarray[0:2])
    i3 = sum(q_idealcountarray[0:3])
    q_thresholdindexarray = np.array([i1, i2, i3])
    return q_thresholdindexarray

def getQuartileThresholdValues(df, q_thresholdindexarray):
    """Get the values for the last item in the 1st, 2nd and 3rd quartiles,
    based on the ideal count arrays"""
    value_array = df['VALUE'].to_numpy()
    p25 = float(df[df.ROW_REF == q_thresholdindexarray[0]]['VALUE'])
    p50 = float(df[df.ROW_REF == q_thresholdindexarray[1]]['VALUE'])
    p75 = float(df[df.ROW_REF == q_thresholdindexarray[2]]['VALUE'])
    q_thresholdarray = np.array([p25, p50, p75])
    return q_thresholdarray

def getQuartileThresholdRealValues(df):
    """Get the values for the 1st, 2nd and 3rd quartiles, based on the overall
    data (there may be some linear interpolation)"""
    value_array = df['VALUE'].to_numpy()
    p25 = np.quantile(value_array, 0.25)
    p50 = np.quantile(value_array, 0.5)
    p75 = np.quantile(value_array, 0.75)
    q_thresholdarray = np.array([p25, p50, p75])
    return q_thresholdarray

def getQuartileCounts(df, q_thresholdarray):
    """Get the number of items allocated to each quartile bucket based on the
    quartile threshold values"""
    value_array = df['VALUE'].to_numpy()
    q1 = sum(value_array <= q_thresholdarray[0])
    q2 = sum(value_array <= q_thresholdarray[1]) - q1
    q3 = sum(value_array <= q_thresholdarray[2]) - q1 - q2
    q4 = len(df) - q1 - q2 - q3
    q_countarray = np.array([q1, q2, q3, q4])
    return q_countarray

def getRepeatedValuesOnBorderDataFrames(df, q_thresholdarray):
    """Returns the data frames containing the data points repeated values at
    the border of each quartile"""
    q1q2_df = df[df.VALUE == q_thresholdarray[0]]
    q2q3_df = df[df.VALUE == q_thresholdarray[1]]
    q3q4_df = df[df.VALUE == q_thresholdarray[2]]
    return [q1q2_df, q2q3_df, q3q4_df]

def getRowAllocationStats(df, thresholdindex):
    """Returns a dictionary of the statistics needed to compute the new row
    references"""
    count = len(df)
    minrowref = min(df['ROW_REF'])
    maxrowref = max(df['ROW_REF'])
    uppercount = maxrowref - thresholdindex
    lowercount = count - uppercount
    ratio = df['CLASS'].value_counts(normalize=True)

    class_labels = ratio.index.sort_values()
    class0_label = class_labels[0]
    class1_label = class_labels[1]
    class0_pct = ratio[class0_label]
    class1_pct = ratio[class1_label]
    # class0_label = ratio.index[0]
    # class1_label = ratio.index[1]
    # class0_pct = ratio[0]
    # class1_pct = ratio[1]
    lowercount_class0 = round(class0_pct * lowercount)
    lowercount_class1 = lowercount - lowercount_class0
    count_class0 = int(round(class0_pct * count))
    count_class1 = int(round(class1_pct * count))
    uppercount_class0 = count_class0 - lowercount_class0
    uppercount_class1 = count_class1 - lowercount_class1
    calc_dict = {'count': count,
                 'count_class0': count_class0, 'count_class1': count_class1,
                 'minrowref': minrowref, 'maxrowref': maxrowref,
                 'uppercount': uppercount, 'lowercount': lowercount,
                 'class0_label': class0_label, 'class1_label': class1_label,
                 'class0_pct': class0_pct, 'class1_pct': class1_pct,
                 'lowercount_class0': lowercount_class0,
                 'lowercount_class1': lowercount_class1,
                 'uppercount_class0': uppercount_class0,
                 'uppercount_class1': uppercount_class1}
    return calc_dict

def reallocateRowRef(df, stats_dict):
    """Creates a dataframe containing the new column indices for the repeated
    values at the border"""

    row_ref = stats_dict['minrowref']
    lowercount_class0_max = stats_dict['lowercount_class0']
    lowercount_class1_max = stats_dict['lowercount_class1']
    class0 = stats_dict['class0_label']
    class1 = stats_dict['class1_label']
    class0_counter = 0
    class1_counter = 0

    for index, row in df.iterrows():
        if row['CLASS'] == class0 and class0_counter < lowercount_class0_max:
            class0_counter += 1
            df.loc[index, 'ROW_REF2'] = row_ref
            row_ref += 1
        elif row['CLASS'] == class1 and class1_counter < lowercount_class1_max:
            class1_counter += 1
            df.loc[index, 'ROW_REF2'] = row_ref
            row_ref += 1
        elif (class0_counter == lowercount_class0_max and
        class1_counter == lowercount_class1_max):
            break

    for index,row in df.iterrows():
        if np.isnan(row['ROW_REF2']):
            df.loc[index, 'ROW_REF2'] = row_ref
            row_ref += 1

    change_count = sum(df['ROW_REF2'] != df['ROW_REF'])
    checksum =  sum(df['ROW_REF2']) - sum(df['ROW_REF'])
    if checksum == 0:
        print("Checksum complete, reallocated {} rows".format(change_count))
    else:
        print("Potential error in re-allocations - please check")
    return df

def updateRowReference(main_df, ref_df_list, q_thresholdindexarray):
    """Merges the results of the row reference reallocation to the main data
    frame"""

    for df in ref_df_list:
        main_df = main_df.merge(df['ROW_REF2'], how='left'
                                , left_index=True, right_index=True)

    main_df['ROW_REF2_COMBINED'] = (main_df['ROW_REF2'].fillna(0) + main_df['ROW_REF2_x'].fillna(0) + main_df['ROW_REF2_y'].fillna(0))
    main_df['ROW_REF_FINAL'] = (main_df.apply(lambda row: row['ROW_REF'] if row['ROW_REF2_COMBINED'] == 0 else row['ROW_REF2_COMBINED'], axis=1))
    main_df['QUARTILE'] = (main_df['ROW_REF_FINAL'].apply(quartileAssignment,i = q_thresholdindexarray))
    main_df = main_df.drop(columns=['ROW_REF2', 'ROW_REF2_x', 'ROW_REF2_y','ROW_REF2_COMBINED', 'ROW_REF'])

    change_count = sum(main_df['ROW_REF'] != main_df['ROW_REF_FINAL'])
    checksum =  sum(main_df['ROW_REF']) - sum(main_df['ROW_REF_FINAL'])
    if checksum == 0:
        print("Final checksum complete, reallocated {} rows".format(change_count))
    else:
        print("Potential error in re-allocations - please check")
    return main_df

def quartileAssignment(x,i):
    i1 = i[0]
    i2 = i[1]
    i3 = i[2]
    if x <= i1:
        label = 'Q1'
    elif x <= i2:
        label = 'Q2'
    elif x <= i3:
        label = 'Q3'
    elif x > i3:
        label = 'Q4'
    return label


def visualiseScatterData(df):
    """Create a simple scatter plot of all the data points"""

    source = ColumnDataSource(data=dict(index=df['ROW_REF_FINAL'],
        value=df['VALUE']))

    # TOOLTIPS = [
    #     ("index", "@index"),
    #     ("value", "@value{0,0}")
    #     ]

    p = figure(plot_width=800, plot_height=400,
        title="Dotplot of values")

    p.circle("index", "value", source=source, size=5,
                fill_color="grey", line_color=None)

    # p.add_tools(HoverTool(tooltips=TOOLTIPS))

    # output_file("line.html")
    # show(p)
    return p

def getReallocationStatsDataFrame(stats_dict_list):
    """Create the dataframe to show the splits at the upper/lower boundaries"""

    df_list = []
    for item in zip(stats_dict_list, ['Q1/Q2', 'Q2/Q3', 'Q3/Q4']):
        stats_dict = item[0]
        group_label = item[1]
        grouping_list = [group_label] * 4
        lowerupper_list = ['lower'] * 2 + ['upper'] * 2
        class_list = [stats_dict['class0_label'], stats_dict['class1_label']]*2
        value_list = [stats_dict['lowercount_class0'],
            stats_dict['lowercount_class1'], stats_dict['uppercount_class0'],
            stats_dict['uppercount_class1'],]

        pct1 = (stats_dict['lowercount_class0']/stats_dict['lowercount']*1.0
            if stats_dict['lowercount'] !=0 else 0)
        pct2 = (stats_dict['lowercount_class1']/stats_dict['lowercount']*1.0
            if stats_dict['lowercount'] !=0 else 0)
        pct3 = (stats_dict['uppercount_class0']/stats_dict['uppercount']*1.0
            if stats_dict['uppercount'] !=0 else 0)
        pct4 = (stats_dict['uppercount_class1']/stats_dict['uppercount']*1.0
            if stats_dict['uppercount'] !=0 else 0)
        pct_list = [pct1, pct2, pct3, pct4]

        df_dict = {'Grouping': grouping_list, "Lower_Upper": lowerupper_list,
            "Class": class_list, "Count": value_list, "Percentage": pct_list}
        df = pd.DataFrame(df_dict)
        df_list.append(df)

    final_stats_df = pd.concat(df_list, axis=0)
    return final_stats_df

def visualiseReallocationStats(stats_df):
    """Create a simple visual of the stats"""

    factors = list(zip(stats_df['Grouping'], stats_df['Lower_Upper']))
    factors = sorted(set(factors), key=factors.index)
    classes = stats_df['Class'].unique().tolist()

    a_dict = {}
    a_dict['x'] = factors
    a_dict[classes[0]] = stats_df[stats_df.Class == classes[0]]['Percentage'].tolist()
    a_dict[classes[1]] = stats_df[stats_df.Class == classes[1]]['Percentage'].tolist()
    a_dict['class0_lbl_format'] = ["{0:.2f}%".format(item*100)
        if item !=0 else '' for item
        in stats_df[stats_df.Class == classes[0]]['Percentage'].tolist()]
    a_dict['class1_lbl_format'] = ["{0:.2f}%".format(item*100)
        if item !=0 else '' for item
        in stats_df[stats_df.Class == classes[1]]['Percentage'].tolist()]
    a_dict['class0_lbl_pos'] = (np.array(a_dict[classes[0]])/2).tolist()
    a_dict['class1_lbl_pos'] = (np.array(a_dict[classes[0]])
        + np.array(a_dict[classes[1]])/2).tolist()

    source = ColumnDataSource(data=a_dict)

    p = figure(y_range=FactorRange(*factors), plot_height=400, plot_width=600,
        toolbar_location=None, tools="",
        title="Distribution of classes at threshold")

    p.hbar_stack(classes, y='x', height=0.9, alpha=0.5, color=["blue", "red"],
        source=source, legend_label=classes)

    p.text('class0_lbl_pos', y='x', source=source, text='class0_lbl_format',
        text_align='center', y_offset=8)
    p.text('class1_lbl_pos', y='x', source=source, text='class1_lbl_format',
        text_align='center', y_offset=8)

    p.add_layout(p.legend[0], 'below')

    p.x_range.start = 0
    p.x_range.end = 1
    p.y_range.range_padding = 0.1
    p.yaxis.major_label_orientation = 1
    p.ygrid.grid_line_color = None
    p.legend.location = "center"
    p.legend.orientation = "horizontal"

    # output_file("bars.html")
    # show(p)

    return p


def calculateQuartileComposition(df):
    """Based on the re-assigned indices, computes the quartile statistics"""
    summary_df_count = df[['CLASS','QUARTILE','ROW_REF_FINAL']].groupby(['QUARTILE','CLASS']).agg(['count'])
    summary_df_pct = summary_df_count.groupby(level=0).apply(lambda x: 100 * x / float(x.sum())).reset_index()
    summary_df_count = summary_df_count.reset_index()
    summary_df_count.columns = ['QUARTILE','CLASS','COUNT']
    summary_df_pct.columns = ['QUARTILE','CLASS','PCT']
    return summary_df_count, summary_df_pct


def runPipeline(file_name):
    data = readData(file_name)
    data = sortIndex(data)
    q_idealcountarray = allocateIdealQuartileCount(data)
    q_thresholdindexarray = getQuartileThresholdIndex(q_idealcountarray)
    q_thresholdarray = getQuartileThresholdValues(data, q_thresholdindexarray)
    q_thresholdarray_real = getQuartileThresholdRealValues(data)
    q_countarray = getQuartileCounts(data, q_thresholdarray)
    df_ref_list = getRepeatedValuesOnBorderDataFrames(data, q_thresholdarray)
    df_ref_list_updated = []
    stats_dict_list = []
    for item in zip(df_ref_list, q_thresholdindexarray):
        stats = getRowAllocationStats(item[0], item[1])
        df_ref = reallocateRowRef(item[0], stats)
        df_ref_list_updated.append(df_ref)
        stats_dict_list.append(stats)
    data = updateRowReference(data, df_ref_list_updated, q_thresholdindexarray)
    v1 = visualiseScatterData(data)
    stats_df = getReallocationStatsDataFrame(stats_dict_list)
    v2 = visualiseReallocationStats(stats_df)
    summary_df_count, summary_df_pct = calculateQuartileComposition(data)


#runPipeline('texastribune.csv')

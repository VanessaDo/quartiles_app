# Quartiles App

import streamlit as st
import base64
import pandas as pd
import numpy as np
import quartiles_base as qb

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="myfilename.csv">Download csv file</a>'
    return href

st.title("Quartiles Grouping App")

"""
### About
For continuous data with binary class labels, this app calculates the class
composition of the data when it is grouped by quartiles. It is intended to be
used as a "sense check" for your own calculations.

Your uploaded file should be in csv format and contain a column labelled 'CLASS'
and a column labelled 'VALUE'.
- The 'CLASS' column should contain two classes (can be labelled in any way you prefer e.g. 'M'/'F', 'Male'/'Female')
- The 'VALUE' column should contain numbers (with no missing values).

NOTE: Any files uploaded with this app are not saved or stored in any location.
"""

"""### Upload File """
uploaded_file = st.file_uploader("Choose a csv file to upload.")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("File Preview (displaying first 10 rows)")
    st.write(df[0:10])

    st.markdown("### Preliminary Calculations")
    data = qb.sortIndex(df)
    st.write('Data count: ', len(data))

    st.markdown("Quartile counts (evenly distributed):""")
    q_idealcountarray = qb.allocateIdealQuartileCount(data)
    q_idealcountarray_df = pd.DataFrame(np.reshape(q_idealcountarray, (1,4)),
        columns=['Q1','Q2','Q3','Q4'])
    st.write(q_idealcountarray_df)

    st.markdown("Threshold values (the value of the last item in each quartile)")
    q_thresholdindexarray = qb.getQuartileThresholdIndex(q_idealcountarray)
    q_thresholdarray = qb.getQuartileThresholdValues(data, q_thresholdindexarray)
    q_thresholdarray_df = pd.DataFrame(np.reshape(q_thresholdarray, (1,3)),
        columns=['Q1','Q2','Q3'])
    st.write(q_thresholdarray_df)

    st.markdown("Reallocation statistics")
    q_countarray = qb.getQuartileCounts(data, q_thresholdarray)
    q_countarray_df = pd.DataFrame(np.reshape(q_countarray, (1,4)), columns=['Q1','Q2','Q3','Q4'])
    df_ref_list = qb.getRepeatedValuesOnBorderDataFrames(data, q_thresholdarray)
    df_ref_list_updated = []
    stats_dict_list = []
    for item in zip(df_ref_list, q_thresholdindexarray):
        stats = qb.getRowAllocationStats(item[0], item[1])
        df_ref = qb.reallocateRowRef(item[0], stats)
        df_ref_list_updated.append(df_ref)
        stats_dict_list.append(stats)
    data = qb.updateRowReference(data, df_ref_list_updated, q_thresholdindexarray)
    v1 = qb.visualiseScatterData(data)
    stats_df = qb.getReallocationStatsDataFrame(stats_dict_list)
    v2 = qb.visualiseReallocationStats(stats_df)
    summary_df_count, summary_df_pct = qb.calculateQuartileComposition(data)
    stats_dict_df = pd.DataFrame(stats_dict_list, index=['Q1/Q2', 'Q2/Q3', 'Q3/Q4'])
    st.write(stats_dict_df)

    st.markdown("### Visualisation of final allocations")
    st.bokeh_chart(v1, use_container_width=True)
    st.bokeh_chart(v2, use_container_width=True)
    st.write(stats_df)

    st.markdown("### Final Quartile Composition by Percentage and Count")
    st.write(summary_df_count)
    st.write(summary_df_pct)

    st.markdown("### Download the Results to CSV")
    st.write(data[0:10])
    st.markdown(get_table_download_link(data), unsafe_allow_html=True)

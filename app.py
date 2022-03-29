import streamlit as st
import pandas as pd
from io import BytesIO

from lab5_funcs import pre_calc, generate_scenarios



from io import BytesIO

def to_excel_all_scens(scens_list):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    for i, scen in enumerate(scens_list, 1):
        #python 3.6+
        scen.to_excel(writer, index=False, sheet_name=f'Scenario_{i}')
        #below python 3.6   
        #df.to_excel(writer, index=False,sheet_name='sheetName_{}'.format(i))
    writer.save()
    processed_data = output.getvalue()
    return processed_data


if 'scenarios' not in st.session_state:
	st.session_state.scenarios = None

st.set_page_config(
    page_title='SA Lab 5: Cross-impact method',
    layout='wide'
)

st.write("# SA Lab 5: Cross-impact method")

with st.sidebar.header('1. Upload your XLSX data'):
    uploaded_file = st.sidebar.file_uploader("Upload XLSX file with probabilities from experts", type=["xlsx"])
    st.sidebar.markdown("""
        [Example of required file](https://github.com/ZaikaBohdan/datasetsforlabs/blob/main/sa_lab5_input.xlsx?raw=true)
    """)

if uploaded_file is not None:
    p_prob = pd.read_excel(uploaded_file)
    n_events = p_prob.shape[0]
    n_experts = p_prob.shape[1]-2
    np_prob, c_prob = pre_calc(p_prob, n_experts)

    view_c_prob = c_prob.drop(columns=[f'p(ei/not_e{i})' for i in range(n_events)]+['p(not_ei)'])
    view_c_prob.iloc[:, 1:] = view_c_prob.iloc[:, 1:].round(4).astype(str).replace({'nan': ''})

    
    st.write("## Experts' probabilities for events")
    st.dataframe(p_prob)

    st.write("## Normalized and generated conditional probabilities for events")
    st.dataframe(view_c_prob)

    with st.sidebar.header('2. Generate scenarios'):
        N_str = st.sidebar.selectbox(
            'N iterations (Monte-Karlo method)',
            ('100', '1000', '10000'),
            2
        )
        gen_button = st.sidebar.button('Generate scenarios')
        
    with st.sidebar: 
        test = st.empty()
        if gen_button:
            test.empty()
            if N_str=='10000':
                test.info('Please, wait. Running 10000 iterations may take about 30 seconds. You will receive message "Done!", when algorithm is finished.')
            with st.spinner():
                st.session_state.scenarios = generate_scenarios(p_prob, n_experts, n_events, int(N_str))
            test.empty()
            test.success('Done! Scroll down to see the results.')

    if st.session_state.scenarios is not None:
        st.write("## Generated scenarios")
        select_scen = st.selectbox(
            'Select Scenario',
            [f'Scenario №{i+1}' for i in range(n_events)],
            0
        )
        n_scen =  int(select_scen[-1]) - 1
        st.dataframe(st.session_state.scenarios[n_scen])
        xlsx_file = to_excel_all_scens(st.session_state.scenarios)
        st.download_button(label='📥 Download Current Results',
                                data=xlsx_file ,
                                file_name= 'generated_scenarios.xlsx')

    else:
        st.info("Choose the number of iterations for Monte-Karlo method and press 'Generate scenarios' button in sidebar.")
            
else:
    st.info('Awaiting for XLSX file to be uploaded in sidebar.')
    st.session_state.scenarios = None
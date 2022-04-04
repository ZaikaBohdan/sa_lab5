from multiprocessing.dummy.connection import families
import streamlit as st
import pandas as pd
from io import BytesIO

from lab5_funcs import pre_calc, generate_scenarios

# Functions
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

def build_report(prior_prob, scens_list, view_cond_prob):
    report = ''
    formater = lambda x: f'{x:.4f}'
    abs_val_list = list()
    p_prob_to_str = prior_prob.to_string(float_format=formater, index=False)
    report += '<===== Початкова імовірність подій за даними експертів =====>\n'
    report += f'{p_prob_to_str}\n\n'
    
    c_prob_to_str = view_cond_prob.to_string(float_format=formater, index=False)
    report += '<===== Нормована імовірність подій та згенеровані умовні ймовірності =====>\n'
    report += f'{c_prob_to_str}\n\n'

    report += '<===== Результати згенерованих сценаріїв =====>\n'
    for i in range(len(scens_list)):
        report += f'Сценарій {i+1}\n'
        df_to_str = scens_list[i].to_string(float_format=formater, index=False)
        report += f'{df_to_str}\n'
        report += f'Підвищення імовірності події e_{i} "{prior_prob.iloc[i, 1]}" до 1 призвело до наступних результатів:\n'
        val_counter = 0
        for j in range(len(scens_list)):
            if j == i:
                continue
            report += f'\t> імовірність події e_{j} "{prior_prob.iloc[j, 1]}" '
            val = round(scens_list[i].loc[j,'Difference'] * 100, 2)
            abs_val = abs(val)
            val_counter += abs_val
            if val > 0:
                report += f'підвищилась на ⇑ {abs_val}%'
            elif val < 0:
                report += f'знизилась на ⇓ {abs_val}%'
            else:
                report += f'не змінилась'
            report += '\n'
        abs_val_list.append((i, round(val_counter,2)))
        report += '\n'
    
    report += 'Рейтинг подій за впливом на систему:\n'
    abs_val_list = sorted(abs_val_list, reverse=True, key=lambda x: x[1])
    for i, el in zip(range(len(scens_list)),abs_val_list):
        report += f'\t{i+1}. Подія e_{el[0]} "{prior_prob.iloc[el[0], 1]}" (всього {el[1]}%)\n'
    
    return report
        

# Interface
if 'scenarios' not in st.session_state:
	st.session_state.scenarios = None

st.set_page_config(
    page_title='SA Lab 5: Cross-impact method',
    page_icon='🎓',
    layout='centered'
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
        st.dataframe(st.session_state.scenarios[0][n_scen])
        st.latex(f"L_1 = {st.session_state.scenarios[1]:.4f};L_4 = {st.session_state.scenarios[2]}; D = {st.session_state.scenarios[3]:.4f}.")

        xlsx_file = to_excel_all_scens(st.session_state.scenarios[0])
        
        col1, col2 = st.columns(2)
        col1.download_button(
            label='📥 Download .xlsx file with results',
            data=xlsx_file, 
            file_name= 'generated_scenarios.xlsx'
            )
        col2.download_button(
            label='📥 Download .txt file with conclusion',
            data=build_report(p_prob, st.session_state.scenarios[0], view_c_prob), 
            file_name= 'conclusion_ua.txt'
            )
        

    else:
        st.info("Choose the number of iterations for Monte-Karlo method and press 'Generate scenarios' button in sidebar.")
            
else:
    st.info('Awaiting for XLSX file to be uploaded in sidebar.')
    st.session_state.scenarios = None
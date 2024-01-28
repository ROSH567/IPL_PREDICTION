import pandas as pd
import streamlit as st 
import pickle
from PIL import Image



teams=['Chennai Super Kings',
 'Delhi Capitals',
 'Kings XI Punjab',
 'Kolkata Knight Riders',
 'Mumbai Indians',
 'Rajasthan Royals',
 'Royal Challengers Bangalore',
 'Sunrisers Hyderabad']

cities=['Hyderabad', 'Pune', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata',
       'Delhi', 'Rajkot', 'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town',
       'Port Elizabeth', 'Durban', 'Centurion', 'East London',
       'Johannesburg', 'Kimberley', 'Bloemfontein', 'Ahmedabad',
       'Cuttack', 'Nagpur', 'Dharamsala', 'Kochi', 'Visakhapatnam',
       'Raipur', 'Ranchi', 'Abu Dhabi', 'Sharjah', 'Kanpur', 'Mohali',
       'Bengaluru']
pipe=pickle.load(open("C:\\Users\\Rishabh\\Downloads\\Rishabh\\projects ML\\ipl\\pipe.pkl",'rb'))

st.title('IPL Win Predictor')

col0,col1=st.columns(2)
with col0:
    batting_team=st.selectbox('Select the batting team', sorted(teams ))
    
with col1:
    bowling_team=st.selectbox('Select the bowling team', sorted(teams ))
city=st.selectbox('Select the city ', sorted(cities))
target= st.number_input('Target',format=None,step=1)

col3,col4,col5= st.columns(3)

with col3:
    score=st.number_input('Current score ',format=None,step=1)

with col4:
    over=st.number_input('Overs completed ',format=None,step=1)

with col5:
    wickets=st.number_input('wickets fallen ',format=None,step=1)

col13,col14,col15= st.columns(3)
with col14:
    if st.button("predict probability "):
        runs_left=target-score
        balls_left=120-(over*6)
        wickets=10-wickets
        crr=score/over 
        rrr=runs_left*6/(balls_left)
    
        data=pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[city],'runs_left':[runs_left],'balls_left':
                       [balls_left],'wickets':[wickets],'total_runs_y':[target],'curr_rr':[crr],'req_rr':[rrr]})
        res=pipe.predict_proba(data)
        loss=res[0][0]
        win=res[0][1]
    
        st.success(batting_team+' - '+str(round(win*100))+"%")
        st.success(bowling_team+' - '+str(round(loss*100))+"%")

    

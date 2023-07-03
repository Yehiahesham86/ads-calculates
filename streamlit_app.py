#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from category_encoders import OneHotEncoder
from sklearn.metrics import accuracy_score,classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st


# In[2]:


data = pd.read_csv("ad_10000records.csv")


# In[3]:


data.head()


# In[4]:


data.info()


# In[27]:


fig=px.histogram(data_frame=data,x="Age",y="Daily Time Spent on Site",title="Daily Time Spent Per Age")
fig.update_layout(
    xaxis_title="Age of User",
    yaxis_title="Daily Time Spent on Site",
    legend_title="Legend Title",
)
st.plotly_chart(fig,use_container_width=True)


# In[6]:


fig=px.histogram(data_frame=data,
                 x=data.Age,color="Clicked on Ad",title="Clicked on Ads per Age",barmode="group",nbins=15)
fig.update_layout(
    xaxis_title="Age of User",
    yaxis_title="num of users",
)

st.plotly_chart(fig,use_container_width=True)

# In[7]:


country_time_spent=data.groupby('Country').sum()["Daily Time Spent on Site"].sort_values().tail(15).round(1)


# In[8]:


fig=px.histogram(data_frame=country_time_spent,
                 x=country_time_spent.index,y=country_time_spent.values,title="Daily Time Spent Per Country",
                 color=country_time_spent.index)
fig.update_layout(
    xaxis_title="Country",
    yaxis_title="Daily Time Spent on Site",
    legend_title="Countries"
)

st.plotly_chart(fig,use_container_width=True)

# In[9]:


fig=px.bar(data_frame=data["Clicked on Ad"].value_counts(normalize=True).round(3)*100,title="Percent % of Clicked on Ads ",barmode="group",color=["0","1"],text_auto="1")
# fig.update_traces(width=0.2)
fig.update_layout(
    xaxis_title="Clicked on Ad",
    yaxis_title="Percent %",
)
fig.update_traces(textposition='inside', textfont_size=14)

st.plotly_chart(fig,use_container_width=True)

# In[10]:


fig=px.histogram(data_frame=data,x="Gender",color="Clicked on Ad",barmode="group",text_auto="1")
st.plotly_chart(fig,use_container_width=True)


# In[11]:


feature="Clicked on Ad"
X=data.drop(columns=["Ad Topic Line","City","Timestamp",feature])
y=data[feature]


# In[12]:





# In[13]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[14]:


model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    RandomForestClassifier(n_estimators=200,random_state=42,n_jobs=-1,class_weight="balanced")
)


# In[15]:


model.fit(x_train,y_train)


# In[16]:


model.score(x_train,y_train)


# In[17]:


model.score(x_test,y_test)


# In[18]:
country_lst=['Svalbard & Jan Mayen Islands', 'Singapore', 'Guadeloupe',
       'Zambia', 'Qatar', 'Cameroon', 'Turkey', 'French Guiana',
       'Vanuatu', 'Burundi', 'Equatorial Guinea', 'Guinea', 'Hong Kong',
       'Spain', 'Uganda', 'Saint Pierre and Miquelon',
       'Northern Mariana Islands', 'Western Sahara', 'Mexico', 'Rwanda',
       'Liechtenstein', 'Bolivia', 'Indonesia', 'Angola', 'Gabon',
       'Saint Vincent and the Grenadines', 'Hungary', 'Honduras',
       'Denmark', 'Tajikistan', 'Afghanistan', 'Micronesia',
       'French Polynesia', 'Tonga', 'Myanmar', 'Croatia', 'Australia',
       'Algeria', 'Greece', 'Bangladesh', 'Latvia', 'Belgium',
       'Czech Republic', 'Cuba', 'Namibia', 'Madagascar', 'Brazil',
       'Barbados', 'Ghana', 'Netherlands', 'Belize', 'American Samoa',
       'Albania', 'Luxembourg', 'Austria', 'Mongolia', 'Ireland',
       'United States Minor Outlying Islands', 'Iran',
       'United Arab Emirates', 'Cambodia', 'Chile', 'South Africa',
       'Estonia', 'Moldova', 'Netherlands Antilles', 'Turkmenistan',
       'Brunei Darussalam', 'Tanzania', 'Kazakhstan', 'Saudi Arabia',
       'Palau', 'Ukraine', "Lao People's Democratic Republic", 'Kuwait',
       'Switzerland', 'Bahamas', 'Mayotte', 'Congo', 'Montenegro',
       'Korea', 'Bulgaria', 'Serbia', 'Somalia', 'Sri Lanka',
       'Libyan Arab Jamahiriya', 'Eritrea', 'Turks and Caicos Islands',
       'Georgia', 'Ethiopia', 'Chad', 'Poland', 'El Salvador',
       'French Southern Territories', 'United States of America',
       'Senegal', 'Dominican Republic', 'Venezuela', 'Saint Lucia',
       'Norway', 'Taiwan', 'Samoa', 'Zimbabwe', 'Finland', 'Slovenia',
       'Fiji', 'Guernsey', 'Cook Islands', 'Yemen', 'United Kingdom',
       'Togo', 'Niger', 'Puerto Rico', 'Mauritius', 'Jamaica', 'Grenada',
       'Timor-Leste', 'Costa Rica', 'Bouvet Island (Bouvetoya)',
       "Cote d'Ivoire", 'United States Virgin Islands', 'Portugal',
       'Faroe Islands', 'Uruguay', 'Jersey', 'Egypt', 'Canada', 'Lebanon',
       'Cayman Islands', 'Bosnia and Herzegovina', 'Israel', 'Sweden',
       'China', 'Malawi', 'Guam', 'Sao Tome and Principe', 'Tuvalu',
       'Peru', 'Maldives', 'Nepal', 'Kyrgyz Republic', 'Liberia',
       'New Caledonia', 'Macedonia', 'San Marino', 'Saint Helena',
       'Malta', 'Greenland', 'Heard Island and McDonald Islands',
       'Cyprus', 'Gibraltar', 'Anguilla', 'Russian Federation',
       'South Georgia and the South Sandwich Islands', 'New Zealand',
       'Antigua and Barbuda', 'Mali', 'Lesotho', 'Papua New Guinea',
       'Christmas Island', 'Dominica', 'Central African Republic',
       'Burkina Faso', 'Bahrain', 'Italy', 'Uzbekistan', 'Azerbaijan',
       'Palestinian Territory', 'Guatemala', 'Nicaragua', 'Malaysia',
       'Japan', 'Isle of Man', 'Gambia', 'Norfolk Island',
       'Wallis and Futuna', 'Sierra Leone', 'Tokelau', 'Kenya', 'France',
       'Germany', 'Andorra', 'Belarus', 'Niue', 'Tunisia', 'Argentina',
       'Mauritania', 'Syrian Arab Republic', 'Panama', 'Monaco',
       'Seychelles', 'Guyana', 'Thailand', 'Mozambique', 'Paraguay',
       'Guinea-Bissau', 'Saint Barthelemy', 'Iceland', 'Philippines',
       'Reunion', 'Suriname', 'Pitcairn Islands', 'Montserrat', 'Bhutan',
       'Falkland Islands (Malvinas)', 'Haiti', 'Martinique']




# In[19]:

def user_input_features():
    gender = st.sidebar.selectbox("Gender",('Male', 'Female'))
    country = st.sidebar.selectbox('Petal width',(country_lst))
    daily_time_spent_on_site = st.sidebar.slider("Daily Time Spent on Site", 10, 500, 50)
    age = st.sidebar.slider("Age", 5, 80, 30)
    area_income = st.sidebar.slider("Area Income", 1000, 100_000, 50_000)
    daily_internet_usage = st.sidebar.slider("Daily Internet Usage", 50,1000, 500)

    data = {"Daily Time Spent on Site": daily_time_spent_on_site,
            "Age": age,
            "Area Income": area_income,
            "Daily Internet Usage": daily_internet_usage,
            "Gender": gender,
            "Country": country}
    features = pd.DataFrame(data, index=[0])
    return features

input_feature = user_input_features()

st.subheader('User Input')

st.write(input_feature)




# In[20]:

predict=model.predict(input_feature)
if predict==0:
    mesg="  User will not Click Ad"
else :
    mesg = "  User will  Click Ad"
    
st.subheader('Prediction')
st.write(mesg)



st.subheader('Prediction Probability')
st.write(model.predict_proba(input_feature))

# In[22]:


print(classification_report(y_test, model.predict(x_test)))


# In[23]:


ConfusionMatrixDisplay.from_estimator(model,x_test,y_test)








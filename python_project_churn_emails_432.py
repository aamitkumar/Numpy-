#!/usr/bin/env python
# coding: utf-8

# # Python Project - Churn Emails - Dataset

# In[1]:


#Count the Number of Lines


# In[2]:


def number_of_lines():
    path='/cxldata/datasets/project/mbox-short.txt'
    fhand=open(path)
    inp=fhand.read()
    fhand.close()
    count=0
    #lines=inp.split('\n')
    for c in inp:
        if c=='\n':
            count=count+1
    return count
        


# In[3]:


print(number_of_lines())


# In[7]:


#Count the Number of lines start with Subject


# In[8]:


import re
def count_number_of_lines():
    path='/cxldata/datasets/project/mbox-short.txt'
    fhand=open(path)
    count=0
    for line in fhand:
        if re.search('Subject:',line):
            count=count+1
    return count


# In[9]:


print(count_number_of_lines())


# In[10]:


#Find Average Spam Confidence


# In[11]:


def average_spam_confidence():
    path='/cxldata/datasets/project/mbox-short.txt'
    fhand=open(path)
    s=0
    count=0
    for line in fhand:
        if line.startswith('X-DSPAM-Confidence:'):
            newline=line.split(':')
            s=s+float(newline[1])
            count=count+1
    ave=s/count
    return ave
            


# In[12]:


print(average_spam_confidence())


# In[13]:


#Find Which Day of the Week the Email was sent


# In[14]:


def find_email_sent_days():
    path='/cxldata/datasets/project/mbox-short.txt'
    fhand=open(path)
    #inp=fhand.read()
    #fhand.close()
    day_count=dict()
    days=list()
    for line in fhand:
        if  line.startswith("From"):
            #print(line)
            words=line.split(' ')
            #print(words)
            length=len(words)
            if length>2:
                days.append(words[2])
    for day in days:
        day_count[day]=day_count.get(day,0)+1
    return day_count
        
    
    


# In[15]:


print(find_email_sent_days())


# In[16]:


#Count Number of Messages From Each Email Address


# In[17]:


def count_message_from_email():
    path='/cxldata/datasets/project/mbox-short.txt'
    fhand=open(path)
    #inp=fhand.read()
    #fhand.close()
    email=list()
    total_email=dict()
    for line in fhand:
        #print(line)
        if line.startswith("From"):
            words=line.split(' ')
            email.append(words[1])
    for id in email:
        total_email[id]=total_email.get(id,0)+1
    return total_email


# In[18]:


print(count_message_from_email())


# In[19]:


#Count Number of Messages From Each Domain


# In[20]:


def count_message_from_domain():
    path='/cxldata/datasets/project/mbox-short.txt'
    fhand=open(path)
    #inp=fhand.read()
    #fhand.close()
    email=list()
    domail=list()
    total_email=dict()
    for line in fhand:
        #print(line)
        if line.startswith("From"):
            words=line.split(' ')
            email.append(words[1])
    for e in email:
        d=e.split('@')
        domail.append(d[1])
    for id in domail:
        total_email[id]=total_email.get(id,0)+1
    return total_email


# In[21]:


print(count_message_from_domain())


# In[ ]:





#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
from urllib.parse import urlparse
from tld import get_tld
from urllib.parse import urlparse
import re

#data = pd.read_csv("data.csv", encoding = "ISO-8859-1", on_bad_lines='skip', sep=';')   --> will be used if importing was faulty
df=pd.read_csv('data.csv')
################################### Hostname Length ###################################
def HostNameLength(url):
    return len(urlparse(url).netloc)

df['host_len'] = df['url'].apply(lambda i: HostNameLength(i))
################################### Count dots in URL ########################################
def dotCount(url):
    dotCount = url.count('.')
    return dotCount

df['dot_count'] = df['url'].apply(lambda i: dotCount(i))
df.head()
################################### Count dots in host name / subTLDs / try to distract the user by Tlds ########################################
def hostDotCount(url):
    count_dot = urlparse(url).netloc.count('.')
    return count_dot

df['host_dot_count'] = df['url'].apply(lambda i: hostDotCount(i))
df.head()
################################### Count www in url / another url in url / redirection !! ###################################
def wwwCount(url):
    url.count('www')
    return url.count('www')

df['www_count'] = df['url'].apply(lambda i: wwwCount(i))
################################### number of embedded URLs ################################### ?????????
def embedUrlCount(url):
    urldir = urlparse(url).path
    return urldir.count('://')-1

df['embed_url_count'] = df['url'].apply(lambda i: embedUrlCount(i))
################################### Count @ in url  ###################################
def atSignCount(url):     
    return url.count('@')

df['count@'] = df['url'].apply(lambda i: atSignCount(i))
################################### number of directories in the URL ###################################
def dirCount(url):
    urldir = urlparse(url).path
    return urldir.count('/')

df['dir_count'] = df['url'].apply(lambda i: dirCount(i))
################################### check if the Url uses shortning services or not from popular services  ###################################
def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',url)
    return 1 if match else 0
        
df['shortened_url'] = df['url'].apply(lambda i: shortening_service(i))
################################### Count http in url / have certification !! ###################################
def httpsCount(url):
    return url.count('https')

df['https_count'] = df['url'].apply(lambda i : httpsCount(i))
###################################  Count http in url / redirection !! ###################################
def httpCount(url):
    return url.count('http')

df['http_count'] = df['url'].apply(lambda i : httpCount(i))

################################### number of spaces in the url! ###################################
def spaceCount(url):
    return url.count('%')

df['space_count'] = df['url'].apply(lambda i : spaceCount(i))
################################### searching for id in the Url ###################################
def questionCount(url):
    return url.count('?')

df['?_count'] = df['url'].apply(lambda i: questionCount(i))
###################################  ###################################
def hyphenCount(url):
    return url.count('-')

df['-_count'] = df['url'].apply(lambda i: hyphenCount(i))
###################################  ###################################
def equalCount(url):
    return url.count('=')

df['=_count'] = df['url'].apply(lambda i: equalCount(i))
df.head()

################################### some suspicious words ###################################
def suspiciousWords(url):
    match = re.search('PayPal|login|signin|bank|account|update|free|lucky|service|bonus|ebayisapi|webscr',
                      url)
    return 1 if match else 0
df['suspicious'] = df['url'].apply(lambda i: suspiciousWords(i))
################################### number of digits ###################################
def digitCount(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
df['digit_count']= df['url'].apply(lambda i: digitCount(i))
################################### number of letters ###################################
def letterCount(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
df['letter_count']= df['url'].apply(lambda i: letterCount(i))
################################### number of special characters ###################################
def specialCount(url):
    letters = 0
    for i in url:
         if not (i.isalpha() or i.isdigit() or i == ' '):
            letters = letters + 1
    return letters

df['special-Count']= df['url'].apply(lambda i: specialCount(i))
df.head()
################################### First Directory Length ###################################
def firstDirLength(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0
df['first_dir_length'] = df['url'].apply(lambda i: firstDirLength(i))
################################### Length of Top Level Domain ###################################
df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))
def tld_length(tld):
    try:
        return len(tld)
    except:
        return -1
df['tld_length'] = df['tld'].apply(lambda i: tld_length(i))

################################### if_com ###################################
df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))
def if_com(tld):
    return 1 if tld == "com" else 0
df['if_com'] = df['tld'].apply(lambda i: if_com(i))
################################### if_org_or_gov ###################################
df['tld'] = df['url'].apply(lambda i: get_tld(i,fail_silently=True))
def if_org_or_gov(tld):
    return 1 if (tld == "org" or tld == "gov") else 0
df['if_org_or_gov'] = df['tld'].apply(lambda i: if_org_or_gov(i))

################################### if_porn_u ###################################
def if_porn_u(url):
    match = re.search("porn",url, flags=re.IGNORECASE)
    return 1 if match else 0
df['if_porn_u'] = df['url'].apply(lambda i: if_porn_u(i))
################################### if_paypal_u ###################################
def if_paypal_u(url):
    match = re.search("paypal",url)
    return 1 if match else 0
df['if_paypal_u'] = df['url'].apply(lambda i: if_paypal_u(i))
################################### if_ali_u ###################################
def if_ali_u(url):
    match = re.search("ali",url)
    return 1 if match else 0
df['if_ali_u'] = df['url'].apply(lambda i: if_ali_u(i))
################################### if_jd_u ###################################
def if_jd_u(url):
    match = re.search("jd",url)
    return 1 if match else 0
df['if_jd_u'] = df['url'].apply(lambda i: if_jd_u(i))
################################### if_safety_u ###################################
def if_safety_u(url):
    match = re.search("safety",url)
    return 1 if match else 0
df['if_safety_u'] = df['url'].apply(lambda i: if_safety_u(i))
################################### if_verify_u ###################################
def if_verify_u(url):
    match = re.search("verify",url)
    return 1 if match else 0
df['if_verify_u'] = df['url'].apply(lambda i: if_verify_u(i))
################################### if_google_u ###################################
def if_google_u(url):
    match = re.search("google",url)
    return 1 if match else 0
df['if_google_u'] = df['url'].apply(lambda i: if_google_u(i))
################################### if_apple_u ###################################
def if_apple_u(url):
    match = re.search("apple",url)
    return 1 if match else 0
df['if_apple_u'] = df['url'].apply(lambda i: if_apple_u(i))
################################### if_facebook_u ###################################
def if_facebook_u(url):
    match = re.search("facebook",url)
    return 1 if match else 0
df['if_facebook_u'] = df['url'].apply(lambda i: if_facebook_u(i))


################################### if_amazon_u ###################################
def if_amazon_u(url):
    match = re.search("amazon",url)
    return 1 if match else 0
df['if_amazon_u'] = df['url'].apply(lambda i: if_amazon_u(i))
################################### if_gamble_u ###################################
def if_gamble_u(url):
    match = re.search("gamble",url)
    return 1 if match else 0
df['if_gamble_u'] = df['url'].apply(lambda i: if_gamble_u(i))
################################### if_porn_d ###################################
def if_porn_d(url):
    domain= urlparse(url).netloc
    match = re.search("porn",domain)
    return 1 if match else 0
df['if_porn_d'] = df['url'].apply(lambda i: if_porn_d(i))
################################### if_paypal_d ###################################
def if_paypal_d(url):
    domain= urlparse(url).netloc
    match = re.search("paypal",domain)
    return 1 if match else 0
df['if_paypal_d'] = df['url'].apply(lambda i: if_paypal_d(i))
################################### if_taobao_d ###################################
def if_taobao_d(url):
    domain= urlparse(url).netloc
    match = re.search("taobao",domain)
    return 1 if match else 0
df['if_taobao_d'] = df['url'].apply(lambda i: if_taobao_d(i))
################################### if_ali_d ###################################
def if_ali_d(url):
    domain= urlparse(url).netloc
    match = re.search("ali",domain)
    return 1 if match else 0
df['if_ali_d'] = df['url'].apply(lambda i: if_ali_d(i))
################################### if_jd_d ###################################
def if_jd_d(url):
    domain= urlparse(url).netloc
    match = re.search("jd",domain)
    return 1 if match else 0
df['if_jd_d'] = df['url'].apply(lambda i: if_jd_d(i))
################################### if_safety_d ###################################
def if_safety_d(url):
    domain= urlparse(url).netloc
    match = re.search("safety",domain)
    return 1 if match else 0
df['if_safety_d'] = df['url'].apply(lambda i: if_safety_d(i))
################################### if_verify_d ###################################
def if_verify_d(url):
    domain= urlparse(url).netloc
    match = re.search("verify",domain)
    return 1 if match else 0
df['if_verify_d'] = df['url'].apply(lambda i: if_verify_d(i))
################################### if_google_d ###################################
def if_google_d(url):
    domain= urlparse(url).netloc
    match = re.search("google",domain)
    return 1 if match else 0
df['if_google_d'] = df['url'].apply(lambda i: if_google_d(i))
################################### if_apple_d ###################################
def if_apple_d(url):
    domain= urlparse(url).netloc
    match = re.search("apple",domain)
    return 1 if match else 0
df['if_apple_d'] = df['url'].apply(lambda i: if_apple_d(i))
################################### if_facebook_d ###################################
def if_facebook_d(url):
    domain= urlparse(url).netloc
    match = re.search("facebook",domain)
    return 1 if match else 0
df['if_facebook_d'] = df['url'].apply(lambda i: if_facebook_d(i))
################################### if_amazon_d ###################################
def if_amazon_d(url):
    domain= urlparse(url).netloc
    match = re.search("amazon",domain)
    return 1 if match else 0
df['if_amazon_d'] = df['url'].apply(lambda i: if_amazon_d(i))
################################### if_gamble_d ###################################
def if_gamble_d(url):
    domain= urlparse(url).netloc
    match = re.search("gamble",domain)
    return 1 if match else 0
df['if_gamble_d'] = df['url'].apply(lambda i: if_gamble_d(i))
################################### if_usa ###################################
def if_usa(geo_loc):
     match = geo_loc == "United States"
     return 1 if match else 0
df['if_usa'] = df['geo_loc'].apply(lambda i: if_usa(i))
################################### if_cn ###################################
def if_cn(geo_loc):
     match = geo_loc == "China"
     return 1 if match else 0
df['if_cn'] = df['geo_loc'].apply(lambda i: if_cn(i))
################################### if_japan ###################################
def if_japan(geo_loc):
     match = geo_loc == "Japan"
     return 1 if match else 0
df['if_japan'] = df['geo_loc'].apply(lambda i: if_japan(i))
################################### if_aus ###################################
def if_aus(geo_loc):
     match = geo_loc == "Australia"
     return 1 if match else 0
df['if_aus'] = df['geo_loc'].apply(lambda i: if_aus(i))
################################### if_india ###################################
def if_india(geo_loc):
     match = geo_loc == "India"
     return 1 if match else 0
df['if_india'] = df['geo_loc'].apply(lambda i: if_india(i))
################################### if_africa ###################################
def if_africa(geo_loc):
     match = geo_loc == "South Africa"
     return 1 if match else 0
df['if_africa'] = df['geo_loc'].apply(lambda i: if_africa(i))


df = df.drop("tld",1) #wont be needed

################################### Corelation  ###################################
correlation_matrix = df.corr()
newDD = correlation_matrix["label"]
print(correlation_matrix["label"])
newDD.to_csv("corelation.csv")

print(df.head())
print(len(df.columns))
df.to_csv("extracted.csv")
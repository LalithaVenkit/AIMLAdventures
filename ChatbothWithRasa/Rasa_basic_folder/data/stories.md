## complete path
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "kolkata"}
    - slot{"location": "kolkata"}
    - action_check_location
    - slot{"validtier": "True"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_price
* restaurant_search{"price": "mid"}
    - slot{"price": "mid"}
    - action_search_restaurants
    - slot{"location": "kolkata"}
    - slot{"cuisine": "chinese"}
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email" : "xyz@gmail.com"}
	- slot{"email": "xyz@gmail.com"}
	- action_send_email
	- utter_sent_mail
    - export

## complete path 2
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "chittoor"}
    - slot{"location": "chittoor"}
    - action_check_location
    - slot{"validtier": "False"}
* restaurant_search{"location": "erode"}
    - slot{"location": "erode"}
    - action_check_location
    - slot{"validtier": "True"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_price
* restaurant_search{"price": "mid"}
    - slot{"price": "mid"}
    - action_search_restaurants
    - slot{"location": "kolkata"}
    - slot{"cuisine": "chinese"}
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email" : "xyz@iiitbml10.net"}
	- slot{"email": "xyz@iiitbml10.net"}
	- action_send_email
	- utter_sent_mail
    - export

## complete path 2 without mail
* greet
    - utter_greet
* restaurant_search
    - utter_ask_location
* restaurant_search{"location": "chittoor"}
    - slot{"location": "chittoor"}
    - action_check_location
    - slot{"validtier": "False"}
* restaurant_search{"location": "erode"}
    - slot{"location": "erode"}
    - action_check_location
    - slot{"validtier": "True"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "chinese"}
    - slot{"cuisine": "chinese"}
    - utter_ask_price
* restaurant_search{"price": "mid"}
    - slot{"price": "mid"}
    - action_search_restaurants
    - slot{"location": "kolkata"}
    - slot{"cuisine": "chinese"}
	- utter_email_query
* deny
	- utter_goodbye

## location specified and valid
* greet
    - utter_greet
* restaurant_search{"location": "delhi"}
    - slot{"location": "delhi"}
    - action_check_location
    - slot{"validtier": "True"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "american"}
    - slot{"cuisine": "american"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants    
    - slot{"location": "delhi"}
    - slot{"cuisine": "american"}
	- utter_email_query
* deny
    - utter_goodbye
    - export

## location specified but invalid
* greet
    - utter_greet
* restaurant_search{"location": "rishikesh"}
    - slot{"location": "rishikesh"}
    - action_check_location
    - slot{"validtier": "False"}
* restaurant_search{"location": "allahabad"}
    - action_check_location
    - slot{"validtier": "True"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "american"}
    - slot{"cuisine": "american"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants    
    - slot{"location": "delhi"}
    - slot{"cuisine": "american"}
	- utter_email_query
* affirm
	- utter_ask_email	
* give_email{"email" : "personname@companyname.net"}
	- slot{"email": "personname@companyname.net"}
	- action_send_email
	- utter_sent_mail
    - export

## location specified but invalid twice
* greet
    - utter_greet
* restaurant_search{"location": "ambattur"}
    - slot{"location": "ambattur"}
    - action_check_location
    - slot{"validtier": "False"}
* restaurant_search{"location": "howrah"}
    - slot{"location": "howrah"}
    - action_check_location
    - slot{"validtier": "False"}
* restaurant_search{"location": "aligarh"}
    - slot{"location": "aligarh"}
    - action_check_location
    - slot{"validtier": "True"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "american"}
    - slot{"cuisine": "american"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants    
    - slot{"location": "delhi"}
    - slot{"cuisine": "american"}
* affirm
    - utter_goodbye
    - export

## cuisine specified without mail
* greet
    - utter_greet
* restaurant_search{"cuisine": "thai"}
    - slot{"cuisine": "thai"}
    - utter_ask_location
* restaurant_search{"location": "udupi"}
    - slot{"location": "udupi"}
	- slot{"validtier":"False"}
* restaurant_search{"location":"mangalore"}
	- slot{"location":"mangalore"}
	- action_check_location
	- slot{"validtier":"True"}
    - utter_ask_price
* restaurant_search{"price": "mid"}
    - slot{"price": "mid"}
    - action_search_restaurants
    - slot{"location": "mangalore"}
    - slot{"cuisine": "thai"}
	- utter_email_query
* deny
    - utter_goodbye
    - export

## cuisine specified with mail
* greet
    - utter_greet
* restaurant_search{"cuisine": "thai"}
    - slot{"cuisine": "thai"}
    - utter_ask_location
* restaurant_search{"location": "bengaluru"}
    - slot{"location": "bangalore"}
	- action_check_location
	- slot{"validtier":"True"}
    - utter_ask_price
* restaurant_search{"price": "mid"}
    - slot{"price": "mid"}
    - action_search_restaurants
    - slot{"location": "bangalore"}
    - slot{"cuisine": "thai"}
	- utter_email_query
* affirm
	- utter_email_query
* give_email{"email":"aloha@city.co.in"}
	- slot{"email":"aloha@city.co.in"}
	- action_send_email
	- utter_sent_mail
    - export

## price specified valid city without mail
* greet
    - utter_greet
* restaurant_search{"price": "low"}
    - slot{"price": "low"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "mexican"}
    - slot{"cuisine": "mexican"}
    - utter_ask_location
* restaurant_search{"location": "delhi"}
	- action_check_location
	- slot{"validtier":"True"}
    - slot{"location": "delhi"}
    - slot{"price": "low"}
    - action_search_restaurants
    - slot{"cuisine": "mexican"}
* deny
    - utter_goodbye
    - export
	
## invalid location and cuisine specified with mail   
* greet
    - utter_greet
* restaurant_search{"cuisine": "american", "location": "mubaim"}
    - slot{"cuisine": "american"}
    - slot{"location": "mubaim"}
	- action_check_location
	- slot{"validtier" : "False"}
* restaurant_search{"location":"navi mumbai"}
	- slot{"location":"mumbai"}
	- action_check_location
	- slot{"validtier" : "True"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants
    - slot{"cuisine": "american"}
    - slot{"location": "mumbai"}
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email":"rahul@upgrad.com"}
	- slot{"email":"rahul@upgrad.com"}
	- action_send_email
	- utter_sent_mail
    - export

## location and cuisine specified with mail   
* greet
    - utter_greet
* restaurant_search{"cuisine": "italian", "location": "trivandrum"}
    - slot{"cuisine": "italian"}
    - slot{"location": "trivandrum"}
	- action_check_location
	- slot{"validtier" : "True"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants
    - slot{"cuisine": "italian"}
    - slot{"location": "trivandrum"}
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email":"rema@upgrad.edu"}
	- slot{"email":"rema@upgrad.edu"}
	- action_send_email
	- utter_sent_mail
    - export
	
## location and price specified without mail   
* greet
    - utter_greet
* restaurant_search{"cuisine": "thai", "location": "dehradun"}
    - slot{"cuisine": "thai"}
    - slot{"location": "dehradun"}
	- action_check_location
	- slot{"validtier" : "True"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants
    - slot{"cuisine": "thai"}
    - slot{"location": "dehradun"}
	- utter_email_query
* deny
	- utter_goodbye
    - export

## valid location and price specified with mail
* greet
    - utter_greet
* restaurant_search{"location": "ahmedabad", "price" : "mid"}
    - slot{"location": "ahmedabad"}
    - slot{"price": "mid"}
	- action_check_location
	- slot{"validtier":"True"}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "north indian"}
    - slot{"cuisine": "north indian"}
    - action_search_restaurants
    - slot{"location": "ahmedabad"}
    - slot{"price": "mid"}
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email":"noealla@nycuni.net"}
	- slot{"email":"noealla@nycuni.net"}
	- action_send_email
	- utter_sent_mail
    - export

## cuisine and invalid location specified without mail
* greet
    - utter_greet
* restaurant_search{"cuisine": "south indian", "location": "nagarcoil"}
    - slot{"cuisine": "south indian"}
    - slot{"location": "nagarcoil"}
	- action_check_location
	- slot{"validtier":"False"}
* restaurant_search{"location":"gwalior"}
    - slot{"location": "gwalior"}
	- action_check_location
	- slot{"validtier":"True"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants
    - slot{"cuisine": "south indian"}
    - slot{"location": "gwalior"}
	- utter_email_query
* deny
    - utter_goodbye
	
## cuisine and invalid location specified with mail
* greet
    - utter_greet
* restaurant_search{"cuisine": "south indian", "location": "cheni"}
    - slot{"cuisine": "south indian"}
    - slot{"location": "cheni"}
	- action_check_location
	- slot{"validtier":"False"}
* restaurant_search{"location":"chennai"}
	- action_check_location
	- slot{"validtier":"True"}
	- slot{"location":"chennai"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants
    - slot{"cuisine": "south indian"}
    - slot{"location": "chennai"}
	- utter_email_query
* affirm
- utter_ask_email
* give_email{"email":"noealla@nycuni.net"}
	- slot{"email":"noealla@nycuni.net"}
	- action_send_email
	- utter_sent_mail
    - export

## cuisine and location specified with mail
* greet
    - utter_greet
* restaurant_search{"cuisine": "north indian", "location": "trichy"}
    - slot{"cuisine": "north indian"}
    - slot{"location": "trichy"}
	- action_check_location
	- slot{"validtier" : "True"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants
    - slot{"cuisine": "north indian"}
    - slot{"location": "trichy"}
	- utter_email_query
* affirm
- utter_ask_email
* give_email{"email":"noealla@nycuni.net"}
	- slot{"email":"noealla@nycuni.net"}
	- action_send_email
	- utter_sent_mail
    - export
	
## cuisine and location specified without mail
* greet
    - utter_greet
* restaurant_search{"cuisine": "north indian", "location": "trichy"}
    - slot{"cuisine": "north indian"}
    - slot{"location": "trichy"}
	- action_check_location
	- slot{"validtier" : "True"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants
    - slot{"cuisine": "north indian"}
    - slot{"location": "trichy"}
	- utter_email_query
* deny
	- utter_goodbye
    - export

## cuisine and price specified
* greet
    - utter_greet
* restaurant_search{"cuisine": "mexican", "price": "low"}
    - slot{"cuisine": "mexican"}
    - slot{"price": "low"}
    - utter_ask_location
	- action_check_location
	- slot{"validtier" : "True"}
* restaurant_search{"location": "coimbatore"}
    - slot{"location": "coimbatore"}
    - action_search_restaurants
    - slot{"cuisine": "mexican"}
    - slot{"price": "low"}
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email":"emailgoes@herethis.org"}
	- slot{"email":"emailgoes@herethis.org"}
	- action_send_email
	- utter_sent_mail
    - export
	
## cuisine and price specified but no mail
* greet
    - utter_greet
* restaurant_search{"cuisine": "south indian", "price": "low"}
    - slot{"cuisine": "south indian"}
    - slot{"price": "low"}
    - utter_ask_location
	- action_check_location
	- slot{"validtier" : "True"}
* restaurant_search{"location": "hyderabad"}
    - slot{"location": "hyderabad"}
    - action_search_restaurants
    - slot{"cuisine": "south indian"}
    - slot{"price": "low"}
	- utter_email_query
* deny
	- utter_goodbye
    - export

## cuisine and price specified but location invalid
* greet
    - utter_greet
* restaurant_search{"cuisine": "mexican", "price": "low"}
    - slot{"cuisine": "mexican"}
    - slot{"price": "low"}
    - utter_ask_location
* restaurant_search{"location": "imphal"}
    - slot{"location": "imphal"}
	- action_check_location
	- slot{"validtier" : "False"}
* restaurant_search{"location": "coimbatore"}
    - slot{"location": "coimbatore"}
	- action_check_location
	- slot{"validtier" : "True"}
    - action_search_restaurants
    - slot{"cuisine": "mexican"}
    - slot{"price": "low"}
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email":"emailg@heret.org"}
	- slot{"email":"emailg@heret.org"}
	- action_send_email
	- utter_sent_mail
    - export
	
## not so happy now
* greet
    - utter_greet
* restaurant_search{"cuisine": "italian", "location": "alappuzha", "price": "mid"}
    - slot{"cuisine": "italian"}
    - slot{"location": "alappuzha"}
    - slot{"price": "mid"}
	- action_check_location
	- slot{"validtier" : "False"}
* restaurant_search{"location": "gurgaon"}
    - slot{"location": "gurgaon"}
	- action_check_location
	- slot{"validtier" : "True"}
    - action_search_restaurants
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email" : "slamdunk@live.com"}
	- slot{"email": "slamdunk@live.com"}
	- action_send_email
	- utter_sent_mail
	
## not so happy and no mails
* greet
    - utter_greet
* restaurant_search{"cuisine": "italian", "location": "bathinda", "price": "mid"}
    - slot{"cuisine": "italian"}
    - slot{"location": "bathinda"}
    - slot{"price": "mid"}
	- action_check_location
	- slot{"validtier" : "False"}
* restaurant_search{"location": "bhopal"}
    - slot{"location": "bhopal"}
	- action_check_location
	- slot{"validtier" : "True"}
    - action_search_restaurants
	- utter_email_query
* deny
	- utter_goodbye
	
## happy but no mail
* greet
    - utter_greet
* restaurant_search{"cuisine": "italian", "location": "noida", "price": "mid"}
    - slot{"cuisine": "italian"}
    - slot{"location": "noida"}
    - slot{"price": "mid"}
	- action_check_location
	- slot{"validtier" : "True"}
    - action_search_restaurants
	- utter_email_query
* deny
	- utter_goodbye
    
## happy_path
* greet
    - utter_greet
* restaurant_search{"cuisine": "italian", "location": "mumbai", "price": "mid"}
    - slot{"cuisine": "italian"}
    - slot{"location": "mumbai"}
    - slot{"price": "mid"}
	- action_check_location
	- slot{"validtier" : "True"}
    - action_search_restaurants
	- utter_email_query
* affirm
	- utter_ask_email
* give_email{"email" : "random@yahoo.co.in"}
	- slot{"email": "random@yahoo.co.in"}
	- action_send_email
	- utter_sent_mail

## interactive_story_1
* restaurant_search{"location": "Bareilly"}
    - slot{"location": "Bareilly"}
    - action_check_location
    - slot{"validtier": true}
    - utter_ask_cuisine
* restaurant_search{"cuisine": "Italian"}
    - slot{"cuisine": "Italian"}
    - utter_ask_price
* restaurant_search{"price": "high"}
    - slot{"price": "high"}
    - action_search_restaurants
    - slot{"location": "bareilly"}
    - utter_email_query
* deny
    - utter_goodbye

## interactive_story_1
* restaurant_search{"price": "high", "cuisine": "south indian", "location": "Delhi", "email": "merin.t@outlook.com"}
    - slot{"cuisine": "south indian"}
    - slot{"email": "merin.t@outlook.com"}
    - slot{"location": "Delhi"}
    - slot{"price": "high"}
    - action_check_location
    - slot{"validtier": true}
    - action_search_restaurants
    - slot{"location": "delhi"}
    - action_send_email
    - utter_sent_mail
    - utter_goodbye

## interactive_story_1
* greet
    - utter_greet
* restaurant_search{"cuisine": "American"}
    - slot{"cuisine": "American"}
    - utter_ask_location
* restaurant_search{"location": "Ahmedabad"}
    - slot{"location": "Ahmedabad"}
    - action_check_location
    - slot{"validtier": true}
    - utter_ask_price
* restaurant_search{"price": "low"}
    - slot{"price": "low"}
    - action_search_restaurants
    - slot{"location": "ahmedabad"}
    - utter_email_query
* deny
    - utter_goodbye

## interactive_story_1
* restaurant_search{"cuisine": "amerika", "location": "Amravati"}
    - slot{"cuisine": "amerika"}
    - slot{"location": "Amravati"}
    - action_check_location
    - slot{"validtier": true}
    - utter_ask_price
* restaurant_search{"price": "mid"}
    - slot{"price": "mid"}
    - action_search_restaurants
    - slot{"location": "amravati"}
    - utter_email_query
* deny
    - utter_goodbye

## interactive_story_1
* restaurant_search{"cuisine": "italian", "location": "Hubli", "price": "low"}
    - slot{"cuisine": "italian"}
    - slot{"location": "Hubli"}
    - slot{"price": "low"}
    - action_check_location
    - slot{"validtier": false}

## interactive_story_2
* restaurant_search{"cuisine": "italian", "location": "Amravati", "price": "morethan700"}
    - slot{"cuisine": "italian"}
    - slot{"location": "Amravati"}
    - slot{"price": "morethan700"}
    - action_check_location
    - slot{"validtier": true}
    - action_search_restaurants
    - slot{"location": "amravati"}
    - utter_email_query
* deny
    - utter_goodbye

## interactive_story_1
* restaurant_search{"cuisine": "south indian", "location": "Bangalore", "price": "mid"}
    - slot{"cuisine": "south indian"}
    - slot{"location": "Bangalore"}
    - slot{"price": "mid"}
    - action_check_location
    - slot{"validtier": true}
    - action_search_restaurants
    - slot{"location": "bangalore"}
    - utter_email_query
* deny
    - utter_goodbye

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

from rasa_sdk import Action
from rasa_sdk.events import SlotSet
import zomatopy
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
restaurant_details_budget_sorted = []

# Budget specifications
budget_dict = {1: (-1,300) , 2: (300,700), 3: (700,100000)}
budget_to_int = {'low':1, 'mid':2, 'high':3}


class ActionSearchRestaurants(Action):
	def name(self):
		return 'action_search_restaurants'
		
	def run(self, dispatcher, tracker, domain):
		# provide API key and initialise a 'zomato app' object
		config={ "user_key":"666f17c4f47beb16d9120532b0fd7d6f"}
		zomato = zomatopy.initialize_app(config)
		
		#Get location, cuisine and budget from slot
		loc = tracker.get_slot('location').lower()
		cuisine = tracker.get_slot('cuisine').lower()
		budget = tracker.get_slot('price')
		
		# get_location gets the lat-long coordinates of 'loc'
		location_detail=zomato.get_location(loc, 1)
		d1 = json.loads(location_detail)
		lat=d1["location_suggestions"][0]["latitude"]
		lon=d1["location_suggestions"][0]["longitude"]
		
		# cuisines code (used by zomatopy)
		cuisines_dict={'bakery':5,'chinese':25,'cafe':30,'italian':55,'biryani':7,'north indian':50,'south indian':85, 'american':1, 'thai':95, 'vegetarian':308, 'mexican':73}
		
		# store retrieved data as a list of dict
		global restaurant_details_budget_sorted
		restaurant_details_budget_sorted = []
		d = self.perform_restaurant_search(zomato, lat, lon, str(cuisines_dict.get(cuisine)), 0)
		response=""
		i=1
		if d['results_found'] == 0:
			response= "no results"
		else:
			self.filter_search_results(d, budget_dict.get(budget_to_int.get(budget)))
			while (len(restaurant_details_budget_sorted) < 10 or i<5):
				d = self.perform_restaurant_search(zomato, lat, lon, str(cuisines_dict.get(cuisine)), 20*i)
				i = i+1
				self.filter_search_results(d, budget_dict.get(budget_to_int.get(budget)))
				if(len(d) < 20):
					break
			response='Showing you top rated restaurants\n'
			for restaurant in restaurant_details_budget_sorted[0:5]:
				response=response+ restaurant['name']+ " in "+ restaurant['area']+" has been rated " + restaurant['rating'] +". And the average price for two people here is :"+str(restaurant['price'])+"\n"
		dispatcher.utter_message("-----"+response)
		return [SlotSet('location',loc)]
		
	def perform_restaurant_search(self, zomato, lat, lon, cuisine, offset):
		results=zomato.restaurant_search("", lat, lon, cuisine, 20, offset)
		d = json.loads(results)
		return d
	
	def filter_search_results(self, d, budget):
		# Filter resturant_details based on budget specifications
		global restaurant_details_budget_sorted
		#If budget is none. Assume user has no budget preference
		if budget is None:
			for restaurant in d['restaurants']:
				restaurant_details_budget_sorted.append({'name' : restaurant['restaurant']['name'], 'area' : restaurant['restaurant']['location']['address'], 'price': restaurant['restaurant']['average_cost_for_two'], 'rating':restaurant['restaurant']['user_rating']['aggregate_rating']})
		else:
			(budget_min, budget_max) = budget
			for restaurant in d['restaurants']:
				if ( budget_min <= restaurant['restaurant']['average_cost_for_two'] < budget_max ) :
					restaurant_details_budget_sorted.append({'name' : restaurant['restaurant']['name'], 'area' : restaurant['restaurant']['location']['address'], 'price': restaurant['restaurant']['average_cost_for_two'], 'rating':restaurant['restaurant']['user_rating']['aggregate_rating']})

class ActionCheckLocation(Action):
	cities = []
	tier_3 = []
	def __init__(self):
		# Tier 1 and Tier 2 cities
		with open('data\\lookup\\cities.txt') as f:
			ActionCheckLocation.cities = [line.rstrip('\n') for line in f]
		ActionCheckLocation.cities = [city.lower() for city in ActionCheckLocation.cities]
		# Tier 3 cities basic list - to have a different response of we do not serve in the location
		with open('data\\lookup\\tier3.txt') as f:
			ActionCheckLocation.tier_3 = [line.rstrip('\n') for line in f]
		ActionCheckLocation.tier_3 = [city.lower() for city in ActionCheckLocation.tier_3]
		
	def name(self):
		return 'action_check_location'
		
	def run(self, dispatcher, tracker, domain):
		# Get location from slot
		loc = tracker.get_slot('location')
		if loc is not None and loc.lower() in ActionCheckLocation.cities:
			return [SlotSet('validtier', True)]
		elif loc is not None and loc.lower() not in ActionCheckLocation.cities and loc.lower() in ActionCheckLocation.tier_3:
			dispatcher.utter_message("Sorry, We don’t operate in this city. Please specify some other location")
			return [SlotSet('validtier', False)]
		else:
			dispatcher.utter_message("Sorry, didn't find any such location. Can you please tell again")
			return [SlotSet('validtier', False)]

class ActionSendEmail(Action):
	def name(self):
		return 'action_send_email'
		
	def run(self, dispatcher, tracker, domain):
		# Get email from slot
		email = tracker.get_slot('email')
		if email is None:
			dispatcher.utter_message("Sorry, email is not valid. Results could not be sent.")
			return
			
		# Store the message in the response
		response = ''
		global restaurant_details_budget_sorted
		restaurant_details_budget_sorted_10 = restaurant_details_budget_sorted[:10]
		for restaurant in restaurant_details_budget_sorted_10:
			response = response + "Found "+ restaurant['name'] + " in "+ restaurant['area'] +" has been rated " + str(restaurant['rating'])+ ". And the average budget for two people is Rs. " + str(restaurant['price']) + "." + "\n"
		
		# Email ID and Password
		sender_email = 'lvappauto@gmail.com'
		sender_password = 'lvapp1234T'
		receiver_email = email
			 
		# Subject for the email
		subject = 'Top 10 Restaurants from Foodie'
			
		mail = MIMEMultipart()
		mail['From'] = sender_email
		mail['To'] = receiver_email
		mail['Subject'] = subject
		body = 'Hey,\n' + response+'\nBon Appetit!'
			
		# Turn the body into plain MIMEText objects and add plain-text parts to MIMEMultipart message   
		mail.attach(MIMEText(body,'plain'))
		
		# Create an unsecured SMTP connection and encrypt it using .starttls().   
		server = smtplib.SMTP('smtp.gmail.com',587)
		server.starttls()
		server.login(sender_email, sender_password)
		server.sendmail(sender_email, receiver_email, mail.as_string())
		server.close()

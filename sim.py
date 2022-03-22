
from person import Person
from person import CareContact
from person import Population
from person import PandemicEvent
from person import HypotheticalVisit
from person import Node
from person import ClassContact
from house import House
from house import Venue
from house import Town
from house import Map
import random
import math
import pylab
import Tkinter
import struct
import time
import sys
import pprint
import pickle
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import csv
import os
from collections import OrderedDict
import operator
import itertools
from itertools import izip_longest
import networkx as nx
import copy
# from PIL import ImageTk         
# from PIL import Image
import pdb


class SimPop:
    def __init__ (self, scenario, params):
        self.p = OrderedDict(params)
         ####  SES variables   #####
        self.socialClassShares = []
        self.careNeedShares = []
        self.householdIncomes = []
        self.individualIncomes = []
        self.incomeFrequencies = []
        self.sesPops = []
        self.sesPopsShares = []
        ## Statistical tallies
        self.times = []
        self.pops = []
        self.avgHouseholdSize = []
        self.marriageTally = 0
        self.numMarriages = []
        self.divorceTally = 0
        self.numDivorces = []
        self.totalCareDemand = []
        self.totalCareSupply = []
        self.informalSocialCareSupply = 0
        self.totalHospitalizationCost = 0
        self.hospitalizationCost = []
        self.numTaxpayers = []
        self.totalUnmetNeed = []
        self.shareUnmetNeed = []
        self.totalFamilyCare = []
        self.inHouseInformalCare = 0
        self.totalTaxBurden = []
        self.marriageProp = []
        self.shareLoneParents = []
        self.shareFemaleLoneParents = []
        self.employmentRate = []
        self.shareWorkingHours = []
        self.publicCareProvision = []
        self.publicSocialCare = 0
        self.costPublicSocialCare = 0
        self.grossDomesticProduct = 0
        self.costTaxFreeSocialCare = 0
        self.costTaxFreeChildCare = 0
        self.costPublicChildCare = 0
        self.publicChildCare = 0
        self.sharePublicSocialCare = 0
        self.sharePublicChildCare = 0
        self.totInHouseSupply = 0
        self.totExternalSupply = 0
        self.inHouseCareSupplyRatio = 0
        self.probsRelocation = []
        self.relocationCostFactors = []
        self.incomeFactors = []
        self.stateTaxRevenue = []
        self.totalTaxRevenue = 0
        self.statePensionRevenue = []
        self.totalPensionRevenue = 0
        self.statePensionExpenditure = []
        self.pensionExpenditure = 0
        
    def load_Data(self):

        self.fert_data = np.genfromtxt('babyrate.txt.csv', skip_header=0, delimiter=',')
        self.death_female = np.genfromtxt('deathrate.fem.csv', skip_header=0, delimiter=',')
        self.death_male = np.genfromtxt('deathrate.male.csv', skip_header=0, delimiter=',')
        self.contacts = pd.read_csv('contactMatrix.csv', header=None)
        self.incomeDistribution = np.genfromtxt('incomeDistribution.csv', skip_header=0, delimiter=',')
        self.incomesPercentiles = np.genfromtxt('incomesPercentiles.csv', skip_header=0, delimiter=',')
        self.wealthPercentiles = np.genfromtxt('wealthDistribution.csv', skip_header=0, delimiter=',')
        
    def run(self, seed): # policy, policyParams, 
        
        self.randSeed = seed
        random.seed(self.randSeed)
        np.random.seed(self.randSeed)
        
        self.load_Data()
        
        if self.p['loadSim'] == False:
            self.initializePop()
            self.updateWealth()
     
        
        fromAgentsToIDs = False
        if self.p['loadSim'] == False:
            for self.year in range(int(self.p['startYear']), int(self.p['endYear'])):
               
                self.doOneYear(self.year)
                
            self.createInfoNetworks_R()
            
            self.createVenues()
            
            if self.p['saveSim'] == True:  # If the simualtion needs to be saved at the end of the yearly-cicle sim
                fromAgentsToIDs = True
                self.from_Agents_to_IDs()
                pickle.dump(self.pop, open('save.p', 'wb'))
                pickle.dump(self.map, open('save.m', 'wb'))
                pickle.dump(self.classContactsMatrix, open('save.cm', 'wb'))
                pickle.dump(self.totalInteractions, open('save.tin', 'wb'))
                pickle.dump(self.classesContacts, open('save.ct', 'wb'))
                pickle.dump(self.householdIndexes, open('save.hi', 'wb'))
                pickle.dump(self.housesNotMoreOccupied, open('save.hno', 'wb'))
                
        else:  # In this case, a saved simulation is uploaded
            print 'Loading base simulation.....'
            self.pop = pickle.load(open('save.p', 'rb'))
            self.map = pickle.load(open('save.m', 'rb'))
            self.classContactsMatrix = pickle.load(open('save.cm', 'rb'))
            self.totalInteractions = pickle.load(open('save.tin', 'rb'))
            self.classesContacts = pickle.load(open('save.ct', 'rb'))
            self.householdIndexes = pickle.load(open('save.hi', 'rb'))
            self.housesNotMoreOccupied = pickle.load(open('save.hno', 'rb'))
            # self.totalAge1Age2Income1Contacts = pickle.load(open('save.tc', 'rb'))
            
        if fromAgentsToIDs == True or self.p['loadSim'] == True:
            
            self.from_IDs_to_Agents()   
            fromAgentsToIDs = False
        
        if self.p['loadSim'] == True:
            if self.p['loadNetwork'] == False:
                self.createInfoNetworks_R()
                if self.p['saveNetwork'] == True:
                    self.from_Agents_to_IDs()
                    pickle.dump(self.pop, open('save.p', 'wb'))
                    pickle.dump(self.map, open('save.m', 'wb'))
                    pickle.dump(self.classContactsMatrix, open('save.cm', 'wb'))
                    pickle.dump(self.totalInteractions, open('save.tin', 'wb'))
                    pickle.dump(self.classesContacts, open('save.ct', 'wb'))
                    pickle.dump(self.householdIndexes, open('save.hi', 'wb'))
                    pickle.dump(self.housesNotMoreOccupied, open('save.hno', 'wb'))
                    self.from_IDs_to_Agents()
            else:
                self.pop = pickle.load(open('save.p', 'rb'))
                self.map = pickle.load(open('save.m', 'rb'))
                self.classContactsMatrix = pickle.load(open('save.cm', 'rb'))
                self.totalInteractions = pickle.load(open('save.tin', 'rb'))
                self.classesContacts = pickle.load(open('save.ct', 'rb'))
                self.householdIndexes = pickle.load(open('save.hi', 'rb'))
                self.housesNotMoreOccupied = pickle.load(open('save.hno', 'rb'))
                self.from_IDs_to_Agents()
                
    def from_Agents_to_IDs(self):
        
        for person in self.pop.allPeople:
            
            if person.mother != None:
                person.motherID = person.mother.id
            else:
                person.motherID = -1
            if person.father != None:
                person.fatherID = person.father.id
            else:
                person.fatherID = -1
            person.childrenID = [x.id for x in person.children]
            person.houseID = person.house.id
            person.mother = None
            person.father = None
            person.children = []
            person.house = None
            
            person.socialContacIDs = []
            person.contactWeights = []
            for agent in [x for x in person.socialContacts.nodes() if x != person]:
                person.socialContacIDs.append(agent.id)
                person.contactWeights.append(person.socialContacts[person][agent]['weight'])
            person.socialContacts.clear()
        
        for house in self.map.allHouses:
            house.infoContacIDs = []
            house.contactWeights = []
            for neighbor in [x for x in house.infoNetwork.nodes() if x != house]:
                house.infoContacIDs.append(neighbor.id)
                house.contactWeights.append(house.infoNetwork[house][neighbor]['weight'])
            house.infoNetwork.clear()
            house.occupantsID = [x.id for x in house.occupants]
            house.occupants = []
      
        for town in self.map.towns:
            town.peopleIDs = [x.id for x in town.people]
            town.people = []
        
    def from_IDs_to_Agents(self):
       
        for person in self.pop.allPeople:
            
            if person.motherID != -1:
                person.mother = [x for x in self.pop.allPeople if x.id == person.motherID][0]
            else:
                person.mother = None
            if person.fatherID != -1:
                person.father = [x for x in self.pop.allPeople if x.id == person.fatherID][0]
            else:
                person.father = None
                
            person.children = [x for x in self.pop.allPeople if x.id in person.childrenID]
            person.socialContacts.clear()
            person.socialContacts.add_node(person)
            
            for index in range(len(person.socialContacIDs)):
                contact = [x for x in self.pop.allPeople if x.id == person.socialContacIDs[index]][0]
                contactWeight = person.contactWeights[index]
                person.socialContacts.add_edge(person, contact, weight = contactWeight)
            
        for person in self.pop.allPeople:
            person.house = [x for x in self.map.allHouses if x.id == person.houseID][0]
            if person in self.pop.livingPeople:
                person.house.occupants.append(person)
                
        for house in self.map.allHouses:
            house.infoNetwork.clear()
            house.infoNetwork.add_node(house)
            for index in range(len(house.infoContacIDs)):
                contact = [x for x in self.map.allHouses if x.id == house.infoContacIDs[index]][0]
                contactWeight = house.contactWeights[index]
                
                # print 'Contact weights: ' + str(contactWeight)
                
                house.infoNetwork.add_edge(house, contact, weight = contactWeight)
        
        for town in self.map.towns:
            town.people = [x for x in self.pop.livingPeople if x.id in town.peopleIDs]
                
    def initializePop(self):
        """
        Set up the initial population and the map.
        We may want to do this from scratch, and we may want to do it
        by loading things from a pre-generated file.
        """
        ## First the map, towns, and houses

        if self.p['loadFromFile'] == False:
            self.map = Map(self.p['mapGridXDimension'],
                           self.p['mapGridYDimension'],
                           self.p['townGridDimension'],
                           self.p['cdfHouseClasses'],
                           self.p['ukMap'],
                           self.p['ukClassBias'],
                           self.p['mapDensityModifier'],
                           self.p['lha_1'], self.p['lha_2'], self.p['lha_3'], self.p['lha_4'])
            
            
        else:
            self.map = pickle.load(open("initMap.txt","rb"))
            

        ## Now the people who will live on it

        if self.p['loadFromFile'] == False:
            self.pop = Population(self.p['initialPop'],
                                  self.p['startYear'],
                                  self.p['minStartAge'],
                                  self.p['maxStartAge'],
                                  self.p['workingAge'],
                                  self.p['incomeInitialLevels'],
                                  self.p['incomeFinalLevels'],
                                  self.p['incomeGrowthRate'],
                                  self.p['workDiscountingTime'],
                                  self.p['wageVar'],
                                  self.p['dailyHours'][0])
            ## Now put the people into some houses
            ## They've already been partnered up so put the men in first, then women to follow
            men = [x for x in self.pop.allPeople if x.sex == 'male']

            remainingHouses = []
            remainingHouses.extend(self.map.allHouses)
        
            for man in men:
                # Assign to each men a house close to people with his own SES
                # Assign to each remaining house a desirability index, based on the difference between the 
                # SES of the man and of the SES of nearby houses
                houseDesirabilityIndex = []
                rearrangedAvailableHouses = []
                for town in self.map.towns:
                    availableHouses = [x for x in remainingHouses if x.town == town]
                    occupiedHouses = [x for x in self.map.occupiedHouses if x.town == town]
                    if len(occupiedHouses) > 0:
                        for house in availableHouses:
                            rearrangedAvailableHouses.append(house)
                            affinityIndexes = []
                            for unit in occupiedHouses:
                                distance = self.geoDistance(house, unit)
                                affinity = 1.0/math.exp(self.p['classAffinityExp']*float(abs(unit.occupants[0].classRank-man.classRank)))
                                affinityIndexes.append(affinity/math.pow(distance, self.p['distanceAffinityExp']))
                            # Compute rent factor
                            house.sizeIndex = 2
                            relRent = house.town.LHA[house.sizeIndex]/math.pow((man.classRank+1), self.p['classRentExp'])
                            rentFactor = 1/math.exp(self.p['rentExp']*relRent)
                            houseDesirabilityIndex.append(np.mean(affinityIndexes)*rentFactor)
                            ## Add cost of houses: low SES agents tend to go in low-cost towns
                            
                            
                if sum(houseDesirabilityIndex) > 0:
                    probs = [x/sum(houseDesirabilityIndex) for x in houseDesirabilityIndex]
                    man.house = np.random.choice(rearrangedAvailableHouses, p = probs)
                else:
                    man.house = np.random.choice(remainingHouses)
                man.currentTown = man.house.town
                man.sec = man.house.size  ## This may not always work, assumes house classes = SEC classes!
                self.map.occupiedHouses.append(man.house)            
                remainingHouses.remove(man.house)
                woman = man.partner
                woman.house = man.house
                woman.currentTown = woman.house.town
                woman.sec = man.sec
                man.yearMarried.append(int(self.p['startYear']))
                woman.yearMarried.append(int(self.p['startYear']))
                man.house.occupants.append(man)
                man.house.occupants.append(woman)

        else:
            self.pop = pickle.load(open("initPop.txt","rb"))
                
    def doOneYear(self, year):
        """Run one year of simulated time."""
        
        self.computeClassShares()
        
        self.doDeaths()
        
        self.doCareTransitions_UCN()
        
        self.resetCareVariables_KN()
        
        ### Care needs, supplies and netwworks  ####
        
        # self.computeSocialCareNeeds_W()
    
        # self.computeChildCareNeeds()
        
        self.householdCareSupply()
        
        # self.householdCareNetwork()
        
        ##### For relocation ####
        # self.computeNetCareDemand()
        
        ############################################
            
        self.doAgeTransitions()
        
        self.doBirths()
        
        self.updateIncome()
        
        self.updateWealth_Ind()
      
        self.doSocialTransition()
        
        self.doDivorces()
        
        self.doMarriages()
        
        self.doMovingAround()
        
        self.checkHouseholdsRelocations()
    
    def createInfoNetworks_R(self):
        
        # First, the initial beta is assigned to each household.
        for house in self.map.occupiedHouses:
            house.infoNetwork.clear()
            # house.beta = self.p['initialBeta']

        self.kinshipNetwork()
        
        # Assign income quintiles to agents
        households = [x for x in self.map.occupiedHouses]
        # households.sort(key=operator.attrgetter("incomePerCapita"))
        households.sort(key=operator.attrgetter("householdIncome"))
        for i in range(int(self.p['incomeClasses'])):
            number = int(round(len(households)/(self.p['incomeClasses']-float(i))))
            quintile = households[:number]
            for j in quintile:
                j.incomeQuintile = i
                for agent in j.occupants:
                    agent.incomeQuintile = i
            households = [x for x in households if x not in quintile]
        
        for person in self.pop.livingPeople:
            person.socialContacts.clear()
            person.socialContacts.add_node(person)
        
        # Increase contacts for over-70 people
        for i in range(int(self.p['interactionAgeClasses'])):
            if i <= 12:
                for j in range(int(self.p['interactionAgeClasses'])):
                    if j > 12:
                        self.contacts[i][j] *= self.p['Over65ContactIncrement']
            else:
                for j in range(int(self.p['interactionAgeClasses'])):
                    self.contacts[i][j] *= self.p['Over65ContactIncrement']
        
        # Compute total contacts (for venues' interaction)
        self.totalInteractions = 0
        denContacts = 0
        for z in range(int(self.p['incomeClasses'])):
            denContacts += (1.0/float(self.p['incomeClasses']))*math.pow(self.p['classContactBias'], z)
        self.classesContacts = []
        for i in range(int(self.p['interactionAgeClasses'])): # 
            print 'Empirical contacts total: ' + str(sum(self.contacts[i]))
            totalContacts = sum(self.contacts[i])*self.p['contactCompoundFactor']
            print 'Total contacts: ' + str(totalContacts)
            # Given these total contacts for the class has a whole, find the contacts fro this age group for each income quintile class
            lowClassContacts = totalContacts/denContacts
            for j in range(int(self.p['incomeClasses'])):
                # Given total contact for age class, determine total contact for each SES class in that age class.
                classContacts = float(lowClassContacts*math.pow(self.p['classContactBias'], j))
                # classContacts *= self.p['networkContactsRatio'] # 3.0
                group = [x for x in self.pop.livingPeople if x.interactionAgeClass == i and x.incomeQuintile == j]
                # Total contacts for that age/income group
                classContacts *= len(group)
                self.totalInteractions += int(round(classContacts))
                totAgeContacts = []
                for z in range(int(self.p['interactionAgeClasses'])):
                    # Determine number of contacts WITH each age group
                    ageContacts = int(np.math.ceil(self.contacts[i][z]*int(round(classContacts))))
                    totAgeContacts.append(ageContacts)
                # The interaction info for the age/SES group (i, j) is stored in the ClassContact class.
                newClass = ClassContact(i, j, len(group), int(round(classContacts)), self.contacts[i], totAgeContacts)
                # List of contacts' info for each age/SES group
                self.classesContacts.append(newClass)
        
        # Here, the size of individual social network is determined.
        # It depends on the age and SES of the agents.
        # The effect of age is based on empirical data (age/age contact matrix)
        # The effect of SES is an assumption of the model, represented by the parameter classContactBias
        numFriends = []
        for i in range(int(self.p['interactionAgeClasses'])): # 
            meanFriends = 0
            for j in range(int(self.p['interactionAgeClasses'])):
                deltaAge = abs(i-j)
                meanFriends += (self.contacts[i][j]/np.exp(self.p['ageFriendsBeta']*deltaAge))*self.p['friendsContactsRatio']
           
            # Given these total contacts for the class has a whole, find the contacts fro this age group for each income quintile class
            lowClassContacts = meanFriends/denContacts
            friendsByIncome = []
            for j in range(int(self.p['incomeClasses'])):
                classFriends = int(math.ceil(lowClassContacts*math.pow(self.p['classContactBias'], j)))
                friendsByIncome.append(classFriends)
            numFriends.append(friendsByIncome)
        # Once the average social network size for each age/SES class is determined, the social network size of each agent
        # is determined through a draw from a Poisson distribution
        totFriends = 0
        for agent in self.pop.livingPeople:
            agent.actualFriends = 0
            agent.numFriends = np.random.poisson(numFriends[agent.interactionAgeClass][agent.incomeQuintile])
            totFriends += agent.numFriends
            
        agentsToLink = [x for x in self.pop.livingPeople if x.actualFriends < x.numFriends]
        agentsWithPotentialFriends = [x for x in agentsToLink if len([y for y in agentsToLink if y not in x.socialContacts.nodes()]) > 0]
        
        # Now, the social netwrok of each agent is computed
        while len(agentsWithPotentialFriends) > 1:
            weights = [float(x.numFriends-x.actualFriends) for x in agentsWithPotentialFriends]
            probs = [x/sum(weights) for x in weights]
            if sum(probs) == 0.0:
                print 'Weights: ' + str(weights)
            agent = np.random.choice(agentsWithPotentialFriends, p = probs)
           
            contactsGroup = [x for x in agentsWithPotentialFriends if x not in agent.socialContacts.nodes() and x not in agent.house.occupants]
            if len(contactsGroup) > 0:
                # Age weight: is given from the empirical age/age contact matrix
                # ageWeight = [self.contacts[agent.interactionAgeClass][x.interactionAgeClass] for x in contactsGroup]
                ageDistances = [abs(agent.age-x.age) for x in contactsGroup]
                ageExps = [self.p['ageInteractionBeta']*math.pow(x, self.p['ageInteractionExp']) for x in ageDistances]
                
                # Agents which are already friend
                friends = [x for x in agent.socialContacts.nodes()]
                # Number of friends of potential agents who are also friend of the agent
                commonFriends = [len([y for y in x.socialContacts.nodes() if y in friends]) for x in contactsGroup]
                friendsExp = [self.p['friendsInteractionBeta']*math.pow(float(x), self.p['friendsInteractionExp']) for x in commonFriends]
                
                # Class factor
                classDistances = [abs(x.incomeQuintile-agent.incomeQuintile) for x in contactsGroup]
                classExps = [self.p['classInteractionBeta']*math.pow(x, self.p['classInteractionExp']) for x in classDistances]
                
                # Geographical distance factor
                geoDistances = [self.manhattanDistance(agent.house.town, x.house.town) for x in contactsGroup]
                distanceExps = [self.p['locationInteractionBeta']*math.pow(x, self.p['locationInteractionExp']) for x in geoDistances]
                
                totalExps = [(a+c+d-f) for a, c, d, f in zip(ageExps, classExps, distanceExps, friendsExp)]
                weightsFullMobility = [1.0/math.exp(x) for x in totalExps]
                # interactionRates = [x.contactReductionRate for x in contactsGroup]
                # weights = [w*r for w, r in zip(weightsFullMobility, interactionRates)]
                contactsProbs = [x/sum(weightsFullMobility) for x in weightsFullMobility]
                potentialContact = np.random.choice(contactsGroup, p = contactsProbs)
                
                quintile = potentialContact.incomeQuintile
                contactIndex = contactsGroup.index(potentialContact)
                contactWeight = weightsFullMobility[contactIndex]
                # The weight of the link is given by the probability of the link creation
                agent.socialContacts.add_edge(agent, potentialContact, weight = contactWeight)
                potentialContact.socialContacts.add_edge(potentialContact, agent, weight = contactWeight)
                # Update number of friends
                agent.actualFriends += 1
                potentialContact.actualFriends += 1
                
                if agent.actualFriends >= agent.numFriends or len([x for x in agentsWithPotentialFriends if x not in agent.socialContacts.nodes()]) == 0:
                    agentsWithPotentialFriends.remove(agent)
                if potentialContact.actualFriends >= potentialContact.numFriends or len([x for x in agentsWithPotentialFriends if x not in potentialContact.socialContacts.nodes()]) == 0:
                    agentsWithPotentialFriends.remove(potentialContact)
                
                totFriends -= 2
              
                # The hiusehold info network is also created, with likns' weights which depend on geographical distance between houses
                if potentialContact.house not in agent.house.infoNetwork.nodes():
                    distance = self.manhattanDistance(agent.house.town, potentialContact.house.town)
                    agent.house.infoNetwork.add_edge(agent.house, potentialContact.house, weight = distance)
                    potentialContact.house.infoNetwork.add_edge(potentialContact.house, agent.house, weight = distance)
            else:
                agentsWithPotentialFriends.remove(agent)  
                
    def kinshipNetwork(self):
        
        for house in self.map.occupiedHouses:
            vfNum = 0
            vfDen = 0
            if nx.is_empty(house.infoNetwork) == True:
                house.infoNetwork.add_node(house)
                existingNodes = [house]
            else:
                existingNodes = [x for x in house.infoNetwork.nodes()]
            
            visited = []
            visited.append(house)
            
            household = list(house.occupants)
            
            if self.lockdown == False or self.p['careLockdown'] == False:
            # Distance 1
                for member in household:
                    kinshipDistance = 1
                    if member.father != None:
                        nok = member.father
                        if nok.dead == False and nok not in household and nok.house not in existingNodes:
                            distance = self.manhattanDistance(house.town, nok.house.town)
                            house.infoNetwork.add_edge(house, nok.house, weight = distance)
                            nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                            # To compute the vulnerability Index
                            deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                            vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                            vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                            existingNodes.append(nok.house)
                        nok = member.mother
                        if nok.dead == False and nok not in household and nok.house not in existingNodes:
                            distance = self.manhattanDistance(house.town, nok.house.town)
                            house.infoNetwork.add_edge(house, nok.house, weight = distance)
                            nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                            
                            deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                            vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                            vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                            existingNodes.append(nok.house)
                    for child in member.children:
                        nok = child
                        if nok.dead == False and nok not in household and nok.house not in existingNodes:
                            distance = self.manhattanDistance(house.town, nok.house.town)
                            house.infoNetwork.add_edge(house, nok.house, weight = distance)
                            nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                            deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                            vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                            vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                            existingNodes.append(nok.house)
                            
                # Distance 2
                for member in household:
                    kinshipDistance = 2
                    if member.father != None:
                        if member.father.father != None:
                            nok = member.father.father
                            if nok.dead == False and nok not in household and nok.house not in existingNodes:
                                distance = self.manhattanDistance(house.town, nok.house.town)
                                house.infoNetwork.add_edge(house, nok.house, weight = distance)
                                nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                                deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                                vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                existingNodes.append(nok.house)
                            nok = member.father.mother
                            if nok.dead == False and nok not in household and nok.house not in existingNodes:
                                distance = self.manhattanDistance(house.town, nok.house.town)
                                house.infoNetwork.add_edge(house, nok.house, weight = distance)
                                nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                                deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                                vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                existingNodes.append(nok.house)
                        if member.mother.father != None:
                            nok = member.mother.father
                            if nok.dead == False and nok not in household and nok.house not in existingNodes:
                                distance = self.manhattanDistance(house.town, nok.house.town)
                                house.infoNetwork.add_edge(house, nok.house, weight = distance)
                                nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                                deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                                vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                existingNodes.append(nok.house)
                            nok = member.mother.mother
                            if nok.dead == False and nok not in household and nok.house not in existingNodes:
                                distance = self.manhattanDistance(house.town, nok.house.town)
                                house.infoNetwork.add_edge(house, nok.house, weight = distance)
                                nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                                deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                                vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                existingNodes.append(nok.house)
                        brothers = list(set(member.father.children + member.mother.children))
                        brothers.remove(member)
                        for brother in brothers:
                            nok = brother
                            if nok.dead == False and nok not in household and nok.house not in existingNodes:
                                distance = self.manhattanDistance(house.town, nok.house.town)
                                house.infoNetwork.add_edge(house, nok.house, weight = distance)
                                nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                                deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                                vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                existingNodes.append(nok.house)
                    for child in member.children:
                        for grandchild in child.children:
                            nok = grandchild
                            if nok.dead == False and nok not in household and nok.house not in existingNodes:
                                distance = self.manhattanDistance(house.town, nok.house.town)
                                house.infoNetwork.add_edge(house, nok.house, weight = distance)
                                nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                                deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                                vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                existingNodes.append(nok.house)
                                
                # Distance 3
                for member in household:
                    kinshipDistance = 3
                    uncles = []
                    if member.father != None:
                        if member.father.father != None:
                            uncles = list(set(member.father.father.children + member.father.mother.children))
                            uncles.remove(member.father)
                        if member.mother.father != None:
                            uncles.extend(list(set(member.mother.father.children + member.mother.mother.children)))
                            uncles.remove(member.mother)
                        for uncle in uncles:
                            nok = uncle
                            if nok.dead == False and nok not in household and nok.house not in visited:
                                distance = self.manhattanDistance(house.town, nok.house.town)
                                house.infoNetwork.add_edge(house, nok.house, weight = distance)
                                nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                                deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                                vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                existingNodes.append(nok.house)
                        brothers = list(set(member.father.children + member.mother.children))
                        brothers.remove(member)
                        for brother in brothers:
                            for child in brother.children:
                                nok = child
                                if nok.dead == False and nok not in household and nok.house not in visited:
                                    distance = self.manhattanDistance(house.town, nok.house.town)
                                    house.infoNetwork.add_edge(house, nok.house, weight = distance)
                                    nok.house.infoNetwork.add_edge(nok.house, house, weight = distance)
                                    deltaAge = np.mean([x.age for x in house.occupants])-np.mean([x.age for x in nok.house.occupants])
                                    vfNum += deltaAge*np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                    vfDen += np.exp(-1*self.p['vulnerabilityExp']*kinshipDistance)
                                    existingNodes.append(nok.house)
                                    
            house.vulnerabilityIndex = 0
            if vfDen > 0:
                house.vulnerabilityIndex = vfNum/vfDen
                
    def createVenues(self):
        
        self.householdIndexes = np.random.choice([x.id for x in self.map.occupiedHouses], 20, replace=False)
        
        print 'Occupied houses index: ' + str(self.householdIndexes)
        # pdb.set_trace()
        self.housesNotMoreOccupied = []
        
        for town in self.map.towns:
            town.people = []
            for house in town.houses:
                town.people.extend(house.occupants)
            sizeTown = len(town.people)
            numVenues = int(math.ceil(float(sizeTown)/float(self.p['agentsPerLocation'])))
            
            print 'Town id: ' + str(town.id)
            print 'Town size: ' + str(sizeTown)
            print 'Town venues: ' + str(numVenues)
            
            occupiedSpots = [[i.x, i.y] for i in town.houses]
            for i in range(numVenues):
                newVenue = Venue(town)
                # Determine venue's location
                randomX = random.randint(0, int(self.p['townGridDimension']))
                randomY = random.randint(0, int(self.p['townGridDimension']))
                while [randomX, randomY] in occupiedSpots:
                    randomX = random.randint(0, int(self.p['townGridDimension']))
                    randomY = random.randint(0, int(self.p['townGridDimension']))
                newVenue.x = randomX
                newVenue.y = randomY
                occupiedSpots.append([randomX, randomY])
                town.venues.append(newVenue)
                self.map.allVenues.append(newVenue)
                
    
        
class SimCov:
    """Instantiates a single run of the simulation."""    
    def __init__ (self, scenario, params, folder):
        
        self.p = OrderedDict(params)
        
        self.dataMap = ['town_x', 'town_y', 'x', 'y', 'size', 'unmetNeed'] 
        
        self.dataPyramid = ['year', 'Class Age 0', 'Class Age 1', 'Class Age 2', 'Class Age 3', 'Class Age 4', 'Class Age 5', 'Class Age 6', 'Class Age 7',
                            'Class Age 8', 'Class Age 9', 'Class Age 10', 'Class Age 11', 'Class Age 12', 'Class Age 13', 'Class Age 14', 'Class Age 15',
                            'Class Age 16', 'Class Age 17', 'Class Age 18', 'Class Age 19', 'Class Age 20', 'Class Age 21', 'Class Age 22', 'Class Age 23', 
                            'Class Age 24']
        
        self.houseData = ['year', 'House name', 'size']
        
        self.householdData = ['ID', 'Sex', 'Age', 'Health']
        
        self.log = ['year', 'message']
        
        self.housesOutputs = ['experiencedEvents[0]', 'kindOfEvents[0]', 'infectionEvents[0]', 'betas[0]', 'contactReductionRates[0]', 
                         'isolationFactors[0]', 'individualIsolations[0]', 'neighborIsolations[0]', 'experiencedEvents[1]', 'kindOfEvents[1]', 
                         'infectionEvents[1]', 'betas[1]', 'contactReductionRates[1]', 'isolationFactors[1]', 'individualIsolations[1]', 
                         'neighborIsolations[1]', 'experiencedEvents[2]', 'kindOfEvents[2]', 'infectionEvents[2]', 'betas[2]', 
                         'contactReductionRates[2]', 'isolationFactors[2]', 'individualIsolations[2]', 'neighborIsolations[2]',
                         'experiencedEvents[3]', 'kindOfEvents[3]', 'infectionEvents[3]', 'betas[3]', 'contactReductionRates[3]', 
                         'isolationFactors[3]', 'individualIsolations[3]', 'neighborIsolations[3]',
                         'experiencedEvents[4]', 'kindOfEvents[4]', 'infectionEvents[4]', 'betas[4]', 'contactReductionRates[4]', 
                         'isolationFactors[4]', 'individualIsolations[4]', 'neighborIsolations[4]', 'experiencedEvents[5]', 'kindOfEvents[5]', 
                         'infectionEvents[5]', 'betas[5]', 'contactReductionRates[5]', 'isolationFactors[5]', 'individualIsolations[5]', 
                         'neighborIsolations[5]', 'experiencedEvents[6]', 'kindOfEvents[6]', 'infectionEvents[6]', 'betas[6]', 
                         'contactReductionRates[6]', 'isolationFactors[6]', 'individualIsolations[6]', 'neighborIsolations[6]',
                         'experiencedEvents[7]', 'kindOfEvents[7]', 'infectionEvents[7]', 'betas[7]', 'contactReductionRates[7]', 
                         'isolationFactors[7]', 'individualIsolations[7]', 'neighborIsolations[7]',
                         'experiencedEvents[8]', 'kindOfEvents[8]', 'infectionEvents[8]', 'betas[8]', 'contactReductionRates[8]', 
                         'isolationFactors[8]', 'individualIsolations[8]', 'neighborIsolations[8]', 'experiencedEvents[9]', 'kindOfEvents[9]', 
                         'infectionEvents[9]', 'betas[9]', 'contactReductionRates[9]', 'isolationFactors[9]', 'individualIsolations[9]', 
                         'neighborIsolations[9]', 'experiencedEvents[10]', 'kindOfEvents[10]', 'infectionEvents[10]', 'betas[10]', 
                         'contactReductionRates[10]', 'isolationFactors[10]', 'individualIsolations[10]', 'neighborIsolations[10]',
                         'experiencedEvents[11]', 'kindOfEvents[11]', 'infectionEvents[11]', 'betas[11]', 'contactReductionRates[11]', 
                         'isolationFactors[11]', 'individualIsolations[11]', 'neighborIsolations[11]',
                         'experiencedEvents[12]', 'kindOfEvents[12]', 'infectionEvents[12]', 'betas[12]', 'contactReductionRates[12]', 
                         'isolationFactors[12]', 'individualIsolations[12]', 'neighborIsolations[12]', 'experiencedEvents[13]', 'kindOfEvents[13]', 
                         'infectionEvents[13]', 'betas[13]', 'contactReductionRates[13]', 'isolationFactors[13]', 'individualIsolations[13]', 
                         'neighborIsolations[13]', 'experiencedEvents[14]', 'kindOfEvents[14]', 'infectionEvents[14]', 'betas[14]', 
                         'contactReductionRates[14]', 'isolationFactors[14]', 'individualIsolations[14]', 'neighborIsolations[14]',
                         'experiencedEvents[15]', 'kindOfEvents[15]', 'infectionEvents[15]', 'betas[15]', 'contactReductionRates[15]', 
                         'isolationFactors[15]', 'individualIsolations[15]', 'neighborIsolations[15]',
                         'experiencedEvents[16]', 'kindOfEvents[16]', 'infectionEvents[16]', 'betas[16]', 'contactReductionRates[16]', 
                         'isolationFactors[16]', 'individualIsolations[16]', 'neighborIsolations[16]', 'experiencedEvents[17]', 'kindOfEvents[17]', 
                         'infectionEvents[17]', 'betas[17]', 'contactReductionRates[17]', 'isolationFactors[17]', 'individualIsolations[17]', 
                         'neighborIsolations[17]', 'experiencedEvents[18]', 'kindOfEvents[18]', 'infectionEvents[18]', 'betas[18]', 
                         'contactReductionRates[18]', 'isolationFactors[18]', 'individualIsolations[18]', 'neighborIsolations[18]',
                         'experiencedEvents[19]', 'kindOfEvents[19]', 'infectionEvents[19]', 'betas[19]', 'contactReductionRates[19]', 
                         'isolationFactors[19]', 'individualIsolations[19]', 'neighborIsolations[19]']
        
        self.Outputs = ['day', 'susceptibles', 'exposed', 'infectious', 'recovered', 'totalDeaths', 
                        'hospitalized', 'intubated', 'deaths', 'asymptomatic', 'mildCondition', 'newCases', 'newExposed', 'over70Hospitalized', 
                        'over70Intubated', 
                        'lockdownState', 'q1_infected', 'q1_hospitalized', 'q1_intubated', 'q2_infected', 'q2_hospitalized', 'q2_intubated',
                        'q3_infected', 'q3_hospitalized', 'q3_intubated', 'q4_infected', 'q4_hospitalized', 'q4_intubated', 'q5_infected', 
                        'q5_hospitalized', 'q5_intubated', 'totalHospitalized', 'cumulatedHospitalizations', 
                        'cumulatedICUs', 'hospitalPopulation', 'icuPopulation',
                        'infectedByClass_0', 'infectedByClass_1', 'infectedByClass_2', 'infectedByClass_3', 'infectedByClass_4',
                        'infectedByAge_0', 'infectedByAge_1', 'infectedByAge_2', 'infectedByAge_3', 'infectedByAge_4',
                        'infectedByAge_5', 'infectedByAge_6', 'infectedByAge_7', 'infectedByAge_8',
                        'hospitalizedByClass_0', 'hospitalizedByClass_1', 'hospitalizedByClass_2', 'hospitalizedByClass_3', 'hospitalizedByClass_4',
                        'hospitalizedQuintilesRatio', 'hospitalizedByAge_0', 'hospitalizedByAge_1', 'hospitalizedByAge_2', 'hospitalizedByAge_3', 
                        'hospitalizedByAge_4', 'hospitalizedByAge_5', 'hospitalizedByAge_6', 'hospitalizedByAge_7', 'hospitalizedByAge_8',
                        'intubatedByClass_0', 'intubatedByClass_1', 'intubatedByClass_2', 'intubatedByClass_3', 'intubatedByClass_4',
                        'intubatedQuintilesRatio', 'intubatedByAge_0', 'intubatedByAge_1', 'intubatedByAge_2', 'intubatedByAge_3', 'intubatedByAge_4',
                        'intubatedByAge_5', 'intubatedByAge_6', 'intubatedByAge_7', 'intubatedByAge_8',
                        'symptomaticByClass_0', 'symptomaticByClass_1', 'symptomaticByClass_2', 'symptomaticByClass_3', 'symptomaticByClass_4',
                        'symptomaticByAge_0', 'symptomaticByAge_1', 'symptomaticByAge_2', 'symptomaticByAge_3', 'symptomaticByAge_4',
                        'symptomaticByAge_5', 'symptomaticByAge_6', 'symptomaticByAge_7', 'symptomaticByAge_8',
                        'deathsByClass_0', 'deathsByClass_1', 'deathsByClass_2', 'deathsByClass_3', 'deathsByClass_4',
                        'deathsQuintilesRatio', 'deathsByAge_0', 'deathsByAge_1', 'deathsByAge_2', 'deathsByAge_3', 'deathsByAge_4',
                        'deathsByAge_5', 'deathsByAge_6', 'deathsByAge_7', 'deathsByAge_8',
                        'unmetSocialCareNeed_Q1', 'unmetSocialCareNeed_Q2', 'unmetSocialCareNeed_Q3', 'unmetSocialCareNeed_Q4', 
                        'unmetSocialCareNeed_Q5', 'mostLeastDeprivedRatio', 'totalInformalSocialCare_Q1', 'totalInformalSocialCare_Q2', 
                        'totalInformalSocialCare_Q3', 'totalInformalSocialCare_Q4', 'totalInformalSocialCare_Q5', 'totalSocialCareNeed_Q1', 
                        'totalSocialCareNeed_Q2', 'totalSocialCareNeed_Q3', 'totalSocialCareNeed_Q4', 'totalSocialCareNeed_Q5', 
                        'informalCareRatio', 'careNeedRatio', 'medianIsolationRate', 'meanIsolationRate',
                        'mobilityReduction_Q1', 'mobilityReduction_Q2', 'mobilityReduction_Q3', 'mobilityReduction_Q4', 'mobilityReduction_Q5',
                        'mobilityReduction_A1', 'mobilityReduction_A2', 'mobilityReduction_A3', 'mobilityReduction_A4', 'mobilityReduction_A5',
                        'mobilityReduction_A6', 'mobilityReduction_A7', 'mobilityReduction_A8', 'mobilityReduction_A9', 
                        'totalAttendances', 'cumulatedAbsences', 'periodTests']
#                        'shareHospitalized1To11', 'shareHospitalized12To18', 'shareHospitalized19To39', 'shareHospitalized40To64',
#                        'shareHospitalizedOver65', 'shareIntubated1To11', 'shareIntubated12To18', 'shareIntubated19To39',
#                        'shareIntubated40To64', 'shareIntubatedOver65', 'shareDeaths1To11', 'shareDeaths12To18', 'shareDeaths19To39',
#                        'shareDeaths40To64', 'shareDeathsOver65', 'Age1To11Share_H', 'Age12To18Share_H', 'Age19To39Share_H',
#                        'Age40To64Share_H', 'Over65Share_H', 'Age1To11Share_I', 'Age12To18Share_I', 'Age19To39Share_I',
#                        'Age40To64Share_I', 'Over65Share_I', 
#        
        
        self.outputData = pd.DataFrame()
        # Save initial parametrs into Scenario folder
        self.folder = folder + '/Scenario_' + str(scenario)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        filePath = self.folder + '/scenarioParameters.csv'
        c = params.copy()
        for key, value in c.iteritems():
            if not isinstance(value, list):
                c[key] = [value]
        with open(filePath, "wb") as f:
            csv.writer(f).writerow(c.keys())
            csv.writer(f).writerows(itertools.izip_longest(*c.values()))
        
        ## Upload the simulation
        print 'Uploading the simulation....'
                
        self.pop = pickle.load(open(self.folder + '/Policy_0/save.p', 'rb'))
        self.map = pickle.load(open(self.folder + '/Policy_0/save.m', 'rb'))
        self.infectionFatalityRatio = pickle.load(open(self.folder + '/Policy_0/save.if', 'rb'))
        self.newCasesRatios = pickle.load(open(self.folder + '/Policy_0/save.nr', 'rb'))
        self.maxNewCases = pickle.load(open(self.folder + '/Policy_0/save.n', 'rb'))
        self.classContactsMatrix = pickle.load(open(self.folder + '/Policy_0/save.cm', 'rb'))
        self.totalInteractions = pickle.load(open(self.folder + '/Policy_0/save.tin', 'rb'))
        self.classesContacts = pickle.load(open(self.folder + '/Policy_0/save.ct', 'rb'))
        self.householdIndexes = pickle.load(open(self.folder + '/Policy_0/save.hi', 'rb'))
        self.housesNotMoreOccupied = pickle.load(open(self.folder + '/Policy_0/save.hno', 'rb'))
        
    
        self.probSymptomatic = []
        self.probsHospitalization = []
        self.probsIntensiveCare = []
        self.intubatedFatalityRatio = []
        
        ## Pandemic variables  #####
        self.susceptibles = 0
        self.exposed = 0
        self.infectious = 0
        self.newCases = 0
        self.pastCases = 0
        self.relativeCasesIncrease = 0
        self.venuesAttendances = []
        self.casesByAge = []
        self.probsConditionsByAge = []
        self.varsConditionsByAge = []
        self.venuesAttendantsDist = [0]*int(self.p['isolationRatesNum'])
        self.shareAttendances = [0]*int(self.p['isolationRatesNum'])
        self.newCasesRatio = 0
        
        # Public probabilities
        self.shareHospitalized1To18 = 0 
        self.shareIntubated1To18 = 0
        self.shareDeaths1To18 = 0
        self.shareHospitalized19To39 = 0 
        self.shareIntubated19To39 = 0
        self.shareDeaths19To39 = 0
        self.shareHospitalized40To59 = 0 
        self.shareIntubated40To59 = 0
        self.shareDeaths40To59 = 0
        self.shareHospitalized60To79 = 0 
        self.shareIntubated60To79 = 0
        self.shareDeaths60To79 = 0
        self.shareHospitalized80Over = 0 
        self.shareIntubated80Over = 0
        self.shareDeaths80Over = 0
        
        self.shareHospitalized1To18_var = 0 
        self.shareIntubated1To18_var = 0
        self.shareDeaths1To18_var = 0
        self.shareHospitalized19To39_var = 0 
        self.shareIntubated19To39_var = 0
        self.shareDeaths19To39_var = 0
        self.shareHospitalized40To59_var = 0 
        self.shareIntubated40To59_var = 0
        self.shareDeaths40To59_var = 0
        self.shareHospitalized60To79_var = 0 
        self.shareIntubated60To79_var = 0
        self.shareDeaths60To79_var = 0
        self.shareHospitalized80Over_var = 0 
        self.shareIntubated80Over_var = 0
        self.shareDeaths80Over_var = 0
        
        self.shareHospitalized1To18_TS = [] 
        self.shareIntubated1To18_TS = []
        self.shareDeaths1To18_TS = []
        self.shareHospitalized19To39_TS = [] 
        self.shareIntubated19To39_TS = []
        self.shareDeaths19To39_TS = []
        self.shareHospitalized40To59_TS = [] 
        self.shareIntubated40To59_TS = []
        self.shareDeaths40To59_TS = []
        self.shareHospitalized60To79_TS = [] 
        self.shareIntubated60To79_TS = []
        self.shareDeaths60To79_TS = []
        self.shareHospitalized80Over_TS = [] 
        self.shareIntubated80Over_TS = []
        self.shareDeaths80Over_TS = []
        
        
        self.jamIndex = 0
        self.recovered = 0
        self.hospitalized = 0
        self.intubated = 0
        self.asymptomatic = 0
        self.mildSymptomatic = 0
        self.deathsForCovid = 0
        self.periodTests = 0
        self.policyStartDeaths = 0
        self.newExposed = 0
        self.newInfected = 0
        self.R0 = 0
        self.maxContacts = 0
        # Lockdown indicators
        self.periodFirstHospitalized = -1
        self.periodFirstIntubated = -1
        self.periodFirstDeath = -1
        self.lockdownMaxCases = 0
        self.lockdownDay = -1
        self.lockdown = False
        self.behaviorModuleOn = True
        self.lockdownEnd = -1
        self.recoveryPeriods = []
        self.contactsMatrix = []
        self.ifr = []
        self.probsDeathIntubated = []
        # Statistics
        self.shareConditions = []
        self.infectedByClass = [0]*int(self.p['incomeClasses'])
        self.infectedByAge = [0]*int(self.p['ageClasses'])
        self.deathsByClass = [0]*int(self.p['incomeClasses'])
        self.deathsByAge = [0]*int(self.p['ageClasses'])
        self.hospitalizedByClass = [0]*int(self.p['incomeClasses'])
        self.hospitalizedByAge = [0]*int(self.p['ageClasses'])
        self.intubatedByClass = [0]*int(self.p['incomeClasses'])
        self.intubatedByAge = [0]*int(self.p['ageClasses'])
        self.symptomaticByClass = [0]*int(self.p['incomeClasses'])
        self.symptomaticByAge = [0]*int(self.p['ageClasses'])
        
        self.deathsToInfectedRatio = 0
        self.deathsToInfectedRatios = []
        self.totalHospitalized = 0
        
        
        self.cumulatedHospitalizations = 0
        self.cumulatedICUs = 0
        self.hospitalPopulation = 0
        self.icuPopulation = 0
        self.totalDeaths = 0
        self.totalInfected = 0
        
        self.infected1To18 = 0
        self.infected19To39 = 0
        self.infected40To59 = 0
        self.infected60To79 = 0
        self.infected80Over = 0
        
        self.hospitalized1To18 = 0
        self.hospitalized19To39 = 0
        self.hospitalized40To59 = 0
        self.hospitalized60To79 = 0
        self.hospitalized80Over = 0
    
        self.icu1To18 = 0
        self.icu19To39 = 0
        self.icu40To59 = 0
        self.icu60To79 = 0
        self.icu80Over = 0
        
        self.dead1To18 = 0
        self.dead19To39 = 0
        self.dead40To59 = 0
        self.dead60To79 = 0
        self.dead80Over = 0
        
        self.totalHospitalizedByClass = [0]*int(self.p['incomeClasses'])
        self.totalIntubatedByClass = [0]*int(self.p['incomeClasses'])
        self.totDeathsByClass = [0]*int(self.p['incomeClasses'])
        
        self.exposedPeriod = 0
        self.infectedNotHospitalizedPeriod = 0
        self.recoveredNotHospitalizedPeriod = 0
        self.periodProbInfected = 0
        self.workingShare = 1.0
        self.gdpRatio = 0
        # Age stats
        # self.infectedByAge = [0 for i in range(9)]
        self.totalSymptomaticByAge = [0 for i in range(9)]
        self.totalHospitalizedByAge = [0 for i in range(9)]
        self.totalIntubatedByAge = [0 for i in range(9)]
        self.deathsByage = [0 for i in range(9)]
        
        self.cumulatedProbInfected = 0
        self.cumulatedPrbInfectious = 0
        self.minAttendances = 10000000
        self.totalAttendants = 0
        ## Counters and storage
        self.year = self.p['startYear']
        self.pandemicDay = 0
        self.pyramid = PopPyramid(self.p['num5YearAgeClasses'],
                                  self.p['numCareLevels'])
        self.textUpdateList = []
        
        self.socialCareNetwork = nx.DiGraph()
        self.infectionsNetwork = nx.DiGraph()
        
        
        
        self.cumulatedAbsences = 0
        self.minIsolationRate = 0
        self.maxIsolationRate = 0
        self.medianIsolationRate = 0
        self.meanIsolationRate = 0
        
        self.mobilityReduction_Q1 = 0
        self.mobilityReduction_Q2 = 0
        self.mobilityReduction_Q3 = 0
        self.mobilityReduction_Q4 = 0
        self.mobilityReduction_Q5 = 0
        
        self.mobilityReduction_A1 = 0
        self.mobilityReduction_A2 = 0
        self.mobilityReduction_A3 = 0
        self.mobilityReduction_A4 = 0
        self.mobilityReduction_A5 = 0
        self.mobilityReduction_A6 = 0
        self.mobilityReduction_A7 = 0
        self.mobilityReduction_A8 = 0
        self.mobilityReduction_A9 = 0
        # if self.p['interactiveGraphics']:
#        self.window = Tkinter.Tk()
#        self.canvas = Tkinter.Canvas(self.window,
#                                width=self.p['screenWidth'],
#                                height=self.p['screenHeight'],
#                                background=self.p['bgColour'])


    def run(self, policy, policyParams, seed):
        """Run the simulation from year start to year end."""
        
        
        #pprint.pprint(self.p)
        #raw_input("Happy with these parameters?  Press enter to run.")
        self.randSeed = seed
        random.seed(self.randSeed)
        np.random.seed(self.randSeed)
        
    
        if self.p['interactiveGraphics']:
            self.initializeCanvas()     
            
        # Save policy parameters in Policy folder
        policyFolder = self.folder + '/Policy_' + str(policy)
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        filePath = policyFolder + '/policyParameters.csv'
        c = policyParams.copy()
        for key, value in c.iteritems():
            if not isinstance(value, list):
                c[key] = [value]
        with open(filePath, "wb") as f:
            csv.writer(f).writerow(c.keys())
            csv.writer(f).writerows(itertools.izip_longest(*c.values()))

        startDay = 0
        
        startSim = time.time()
        
        dataHouseholdFolder = os.path.join(policyFolder, 'DataHousehold')
        if not os.path.exists(dataHouseholdFolder):
            os.makedirs(dataHouseholdFolder)
        
        dataMapFolder = os.path.join(policyFolder, 'DataMap')
        if not os.path.exists(dataMapFolder):
            os.makedirs(dataMapFolder)
        
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Log.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.log))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "HouseData.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.houseData))
        
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Male_0.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Male_1.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Male_2.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Male_3.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Male_4.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Female_0.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Female_1.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Female_2.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Female_3.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
            
        if not os.path.exists(policyFolder):
            os.makedirs(policyFolder)
        with open(os.path.join(policyFolder, "Pyramid_Female_4.csv"), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataPyramid))
        
        
        if policy == 0:
            self.setPandemicWeights()
          
        if policy != 0:
            self.lockdownDay = pickle.load(open(self.folder + '/Policy_0/save.l', 'rb'))
            startDay = self.lockdownDay
            
        for self.pandemicDay in range(int(startDay), int(self.p['pandemicPeriod']+1)):
           
            if self.pandemicDay == self.lockdownDay:
                print 'Lockdown started in day: ' + str(self.pandemicDay)
            
            if policyParams and self.pandemicDay == self.lockdownDay:
                
                keys = policyParams.keys()
                for k in keys[1:]:
                    self.p[k] = policyParams[k]
                
                if policy == 0:
                    
                    
                    print 'Saving the simulation....'
                    
                    self.from_Agents_to_IDs()
                    # self.policyStartDeaths = self.totalDeaths
                    # self.lockdownMaxCases = self.maxNewCases
                    # Save outputs
                    self.outputData = pd.read_csv(policyFolder + '/Outputs.csv')
                    self.outputData.to_csv(policyFolder + '/tempOutputs.csv', index=False)
                    # Save simulation
                    pickle.dump(self.pop, open(policyFolder + '/save.p', 'wb'))
                    pickle.dump(self.map, open(policyFolder + '/save.m', 'wb'))
                    pickle.dump(self.infectionFatalityRatio, open(policyFolder + '/save.if', 'wb'))
                    pickle.dump(self.newCasesRatios, open(policyFolder + '/save.nr', 'wb'))
                    pickle.dump(self.maxNewCases, open(policyFolder + '/save.n', 'wb'))
                    pickle.dump(self.classContactsMatrix, open(policyFolder + '/save.cm', 'wb'))
                    pickle.dump(self.totalInteractions, open(policyFolder + '/save.tin', 'wb'))
                    pickle.dump(self.classesContacts, open(policyFolder + '/save.ct', 'wb'))
                    pickle.dump(self.householdIndexes, open(policyFolder + '/save.hi', 'wb'))
                    pickle.dump(self.housesNotMoreOccupied, open(policyFolder + '/save.hno', 'wb'))
                    # pickle.dump(self.totalAge1Age2Income1Contacts, open(policyFolder + '/save.tc', 'wb'))
                    pickle.dump(self.totalDeaths, open(policyFolder + '/save.d', 'wb'))
                    pickle.dump(self.lockdownDay, open(policyFolder + '/save.l', 'wb'))
                    
                    pickle.dump(self.probsDeathIntubated, open(policyFolder + '/save.di', 'wb'))
                    pickle.dump(self.probSymptomatic, open(policyFolder + '/save.sy', 'wb'))
                    pickle.dump(self.probsHospitalization, open(policyFolder + '/save.ho', 'wb'))
                    pickle.dump(self.probsIntensiveCare, open(policyFolder + '/save.ic', 'wb'))
                    pickle.dump(self.intubatedFatalityRatio, open(policyFolder + '/save.vf', 'wb'))
                    pickle.dump(self.recoveryPeriods, open(policyFolder + '/save.rp', 'wb'))
                    
                    # Save the data
                    pickle.dump(self.cumulatedHospitalizations, open(policyFolder + '/save.ch', 'wb'))
                    pickle.dump(self.cumulatedICUs, open(policyFolder + '/save.cv', 'wb'))
                    pickle.dump(self.hospitalPopulation, open(policyFolder + '/save.hp', 'wb'))
                    pickle.dump(self.icuPopulation, open(policyFolder + '/save.vp', 'wb'))
                    pickle.dump(self.totalDeaths, open(policyFolder + '/save.td', 'wb'))
                    pickle.dump(self.totalInfected, open(policyFolder + '/save.ti', 'wb'))
                    pickle.dump(self.totalHospitalizedByClass, open(policyFolder + '/save.hc', 'wb'))
                    pickle.dump(self.totalIntubatedByClass, open(policyFolder + '/save.vc', 'wb'))
                    pickle.dump(self.totDeathsByClass, open(policyFolder + '/save.dc', 'wb'))
                    
                    
                    pickle.dump(self.infected1To18, open(policyFolder + '/save.fa1', 'wb'))
                    pickle.dump(self.infected19To39, open(policyFolder + '/save.fa2', 'wb'))
                    pickle.dump(self.infected40To59, open(policyFolder + '/save.fa3', 'wb'))
                    pickle.dump(self.infected60To79, open(policyFolder + '/save.fa4', 'wb'))
                    pickle.dump(self.infected80Over, open(policyFolder + '/save.fa5', 'wb'))
                    
                    pickle.dump(self.hospitalized1To18, open(policyFolder + '/save.ha1', 'wb'))
                    pickle.dump(self.hospitalized19To39, open(policyFolder + '/save.ha2', 'wb'))
                    pickle.dump(self.hospitalized40To59, open(policyFolder + '/save.ha3', 'wb'))
                    pickle.dump(self.hospitalized60To79, open(policyFolder + '/save.ha4', 'wb'))
                    pickle.dump(self.hospitalized80Over, open(policyFolder + '/save.ha5', 'wb'))
                
                    pickle.dump(self.icu1To18, open(policyFolder + '/save.ia1', 'wb'))
                    pickle.dump(self.icu19To39, open(policyFolder + '/save.ia2', 'wb'))
                    pickle.dump(self.icu40To59, open(policyFolder + '/save.ia3', 'wb'))
                    pickle.dump(self.icu60To79, open(policyFolder + '/save.ia4', 'wb'))
                    pickle.dump(self.icu80Over, open(policyFolder + '/save.ia5', 'wb'))
                    
                    pickle.dump(self.dead1To18, open(policyFolder + '/save.da1', 'wb'))
                    pickle.dump(self.dead19To39, open(policyFolder + '/save.da2', 'wb'))
                    pickle.dump(self.dead40To59, open(policyFolder + '/save.da3', 'wb'))
                    pickle.dump(self.dead60To79, open(policyFolder + '/save.da4', 'wb'))
                    pickle.dump(self.dead80Over, open(policyFolder + '/save.da5', 'wb'))
                    
                    pickle.dump(self.infectionsNetwork, open(policyFolder + '/save.if_nw', 'wb'))
                
                # Upload simulation
                print 'Uploading the simulation....'
                
                self.pop = pickle.load(open(self.folder + '/Policy_0/save.p', 'rb'))
                self.map = pickle.load(open(self.folder + '/Policy_0/save.m', 'rb'))
                self.infectionFatalityRatio = pickle.load(open(self.folder + '/Policy_0/save.if', 'rb'))
                self.newCasesRatios = pickle.load(open(self.folder + '/Policy_0/save.nr', 'rb'))
                self.maxNewCases = pickle.load(open(self.folder + '/Policy_0/save.n', 'rb'))
                self.classContactsMatrix = pickle.load(open(self.folder + '/Policy_0/save.cm', 'rb'))
                self.totalInteractions = pickle.load(open(self.folder + '/Policy_0/save.tin', 'rb'))
                self.classesContacts = pickle.load(open(self.folder + '/Policy_0/save.ct', 'rb'))
                self.householdIndexes = pickle.load(open(self.folder + '/Policy_0/save.hi', 'rb'))
                self.housesNotMoreOccupied = pickle.load(open(self.folder + '/Policy_0/save.hno', 'rb'))
                
                
                # self.totalAge1Age2Income1Contacts = pickle.load(open(self.folder + '/Policy_0/save.tc', 'rb'))
                self.totalDeaths = pickle.load(open(self.folder + '/Policy_0/save.d', 'rb'))
                # self.lockdownDay = pickle.load(open(self.folder + '/Policy_0/save.l', 'rb'))
                
                self.probsDeathIntubated = pickle.load(open(self.folder + '/Policy_0/save.di', 'rb'))
                self.probSymptomatic = pickle.load(open(self.folder + '/Policy_0/save.sy', 'rb'))
                self.probsHospitalization = pickle.load(open(self.folder + '/Policy_0/save.ho', 'rb'))
                self.probsIntensiveCare = pickle.load(open(self.folder + '/Policy_0/save.ic', 'rb'))
                self.intubatedFatalityRatio = pickle.load(open(self.folder + '/Policy_0/save.vf', 'rb'))
                self.recoveryPeriods = pickle.load(open(self.folder + '/Policy_0/save.rp', 'rb'))
                
                self.cumulatedHospitalizations = pickle.load(open(self.folder + '/Policy_0/save.ch', 'rb'))
                self.cumulatedICUs = pickle.load(open(self.folder + '/Policy_0/save.cv', 'rb'))
                self.hospitalPopulation = pickle.load(open(self.folder + '/Policy_0/save.hp', 'rb'))
                self.icuPopulation = pickle.load(open(self.folder + '/Policy_0/save.vp', 'rb'))
                self.totalDeaths = pickle.load(open(self.folder + '/Policy_0/save.td', 'rb'))
                self.totalInfected = pickle.load(open(self.folder + '/Policy_0/save.ti', 'rb'))
                self.totalHospitalizedByClass = pickle.load(open(self.folder + '/Policy_0/save.hc', 'rb'))
                self.totalIntubatedByClass = pickle.load(open(self.folder + '/Policy_0/save.vc', 'rb'))
                self.totDeathsByClass = pickle.load(open(self.folder + '/Policy_0/save.dc', 'rb'))
                
                self.infected1To18 = pickle.load(open(self.folder + '/Policy_0/save.fa1', 'rb'))
                self.infected19To39 = pickle.load(open(self.folder + '/Policy_0/save.fa2', 'rb'))
                self.infected40To59 = pickle.load(open(self.folder + '/Policy_0/save.fa3', 'rb'))
                self.infected60To79 = pickle.load(open(self.folder + '/Policy_0/save.fa4', 'rb'))
                self.infected80Over = pickle.load(open(self.folder + '/Policy_0/save.fa5', 'rb'))
                
                self.hospitalized1To18 = pickle.load(open(self.folder + '/Policy_0/save.ha1', 'rb'))
                self.hospitalized19To39 = pickle.load(open(self.folder + '/Policy_0/save.ha2', 'rb'))
                self.hospitalized40To59 = pickle.load(open(self.folder + '/Policy_0/save.ha3', 'rb'))
                self.hospitalized60To79 = pickle.load(open(self.folder + '/Policy_0/save.ha4', 'rb'))
                self.hospitalized80Over = pickle.load(open(self.folder + '/Policy_0/save.ha5', 'rb'))
            
                self.icu1To18 = pickle.load(open(self.folder + '/Policy_0/save.ia1', 'rb'))
                self.icu19To39 = pickle.load(open(self.folder + '/Policy_0/save.ia2', 'rb'))
                self.icu40To59 = pickle.load(open(self.folder + '/Policy_0/save.ia3', 'rb'))
                self.icu60To79 = pickle.load(open(self.folder + '/Policy_0/save.ia4', 'rb'))
                self.icu80Over = pickle.load(open(self.folder + '/Policy_0/save.ia5', 'rb'))
                
                self.dead1To18 = pickle.load(open(self.folder + '/Policy_0/save.da1', 'rb'))
                self.dead19To39 = pickle.load(open(self.folder + '/Policy_0/save.da2', 'rb'))
                self.dead40To59 = pickle.load(open(self.folder + '/Policy_0/save.da3', 'rb'))
                self.dead60To79 = pickle.load(open(self.folder + '/Policy_0/save.da4', 'rb'))
                self.dead80Over = pickle.load(open(self.folder + '/Policy_0/save.da5', 'rb'))
            
                self.infectionsNetwork = pickle.load(open(self.folder + '/Policy_0/save.if_nw', 'rb'))
                
                print 'Performing from ID to agents 3'
                
                self.from_IDs_to_Agents()
                
                # Upload outputs
                if policy != 0:
                    self.lockdown = self.p['lockdown']
                    self.outputData = pd.read_csv(self.folder + '/Policy_0/tempOutputs.csv')
                    self.outputData.to_csv(policyFolder + '/Outputs.csv', index=False)
                    
            if policy != 0 and self.lockdown == True and self.pandemicDay == startDay + self.p['lockdownDuration']:# self.pandemicDay == self.lockdownEnd:
                self.lockdown = False
                print 'Lockdown ended in day: ' + str(self.pandemicDay)
                    
            print 'Lockdown state: ' + str(self.lockdown)
            print 'Care lockdown state: ' + str(self.p['careLockdown'])
            print 'Max Cases: ' + str(self.maxNewCases)
            print 'Cases at lockdown: ' + str(self.lockdownMaxCases)
            self.doOneDay(self.pandemicDay, policyFolder, dataMapFolder, dataHouseholdFolder)
            
        endSim = time.time()
        
        self.saveInfectionNetwork(policyFolder)
        
        simulationTime = endSim - startSim
        
        print ''
        print 'Simulation time: ' + str(simulationTime)
        
        if self.p['interactiveGraphics']:
            print "Entering main loop to hold graphics up there."
            self.window.mainloop()

    def from_Agents_to_IDs(self):
        
        for person in self.pop.allPeople:
            
            if person.mother != None:
                person.motherID = person.mother.id
            else:
                person.motherID = -1
            if person.father != None:
                person.fatherID = person.father.id
            else:
                person.fatherID = -1
            person.childrenID = [x.id for x in person.children]
            person.houseID = person.house.id
            person.mother = None
            person.father = None
            person.children = []
            person.house = None
            
            person.socialContacIDs = []
            person.contactWeights = []
            for agent in [x for x in person.socialContacts.nodes() if x != person]:
                person.socialContacIDs.append(agent.id)
                person.contactWeights.append(person.socialContacts[person][agent]['weight'])
            person.socialContacts.clear()
        
        for house in self.map.allHouses:
            house.infoContacIDs = []
            house.contactWeights = []
            for neighbor in [x for x in house.infoNetwork.nodes() if x != house]:
                house.infoContacIDs.append(neighbor.id)
                house.contactWeights.append(house.infoNetwork[house][neighbor]['weight'])
            house.infoNetwork.clear()
            house.occupantsID = [x.id for x in house.occupants]
            house.occupants = []
      
        for town in self.map.towns:
            town.peopleIDs = [x.id for x in town.people]
            town.people = []
        
    def from_IDs_to_Agents(self):
       
        for person in self.pop.allPeople:
            
            if person.motherID != -1:
                person.mother = [x for x in self.pop.allPeople if x.id == person.motherID][0]
            else:
                person.mother = None
            if person.fatherID != -1:
                person.father = [x for x in self.pop.allPeople if x.id == person.fatherID][0]
            else:
                person.father = None
                
            person.children = [x for x in self.pop.allPeople if x.id in person.childrenID]
            person.socialContacts.clear()
            person.socialContacts.add_node(person)
            
            for index in range(len(person.socialContacIDs)):
                contact = [x for x in self.pop.allPeople if x.id == person.socialContacIDs[index]][0]
                contactWeight = person.contactWeights[index]
                person.socialContacts.add_edge(person, contact, weight = contactWeight)
            
        for person in self.pop.allPeople:
            person.house = [x for x in self.map.allHouses if x.id == person.houseID][0]
            if person in self.pop.livingPeople:
                person.house.occupants.append(person)
                
        for house in self.map.allHouses:
            house.infoNetwork.clear()
            house.infoNetwork.add_node(house)
            for index in range(len(house.infoContacIDs)):
                contact = [x for x in self.map.allHouses if x.id == house.infoContacIDs[index]][0]
                contactWeight = house.contactWeights[index]
                
                # print 'Contact weights: ' + str(contactWeight)
                
                house.infoNetwork.add_edge(house, contact, weight = contactWeight)
        
        for town in self.map.towns:
            town.people = [x for x in self.pop.livingPeople if x.id in town.peopleIDs]
            
    
    def doOneDay(self, day, policyFolder, dataMapFolder, dataHouseholdFolder):
        
        if day%10 == 0:
            over80 = [x for x in self.pop.livingPeople if x.age >= 80]
            print 'Share of over-80s: ' + str(len(over80))
        
        print 'Day ' + str(day)
        
        ### Pandemic process  #########
        
        print 'Doing symptomsProgression...'
        
        self.socialContactsData()
       
        self.symptomsProgression(day) ## Here the events for the period are recorded
        
        self.updatePublicInfo()
        
        
        # Behavioural module
        self.computeCostOfInfection()
        
        self.updateInfectionRisk()
        
        self.performTest()

        self.updateIsolationRates(day)
        
        
        self.socialNorm()
        
        print 'Hospitalized by class: ' + str(self.totalHospitalizedByClass)
        
        #################################
        
        # self.mobilityRates()
        
        print 'Doing domesticInteraction function...'
        
        self.domesticInteraction()
        
        print 'Doing socialInteraction function...'

        # self.socialInteraction_Debug()
        self.commutingProcess()
        
        self.socialInteraction_SPE_II()
        
        print 'Updating venue distribution'
        
        self.updateVenuesDistribution()
        
        
        print 'Doing exposureProcess....'
        
        self.exposureProcess(day)
        
        print 'Doing stats...'
        
        ################################
        
        self.doStats(day, policyFolder, dataMapFolder, dataHouseholdFolder)
        
        # print 'Doing stats....'
        
    def saveInfectionNetwork(self, policyFolder):
        
        pickle.dump(self.infectionsNetwork, open(policyFolder + '/save.if_nw', 'wb'))
        
        nx.write_gexf(self.infectionsNetwork, policyFolder + '/infectionsNetwork.gexf')
#        edges = self.infectionsNetwork.edges()
#        colors = [self.infectionsNetwork[u][v]['color'] for u,v in edges]
#        nx.draw(self.infectionsNetwork, edges=edges, edge_color=colors)
#        plt.savefig("infectionNetwork.pdf")
#        plt.show()
        
    def socialContactsData(self):
        
        popByAgeCount = [0]*5
        snSizeCount = [0]*5
        popQuintile = [0]*5
        sizeQuintile = [0]*5
        
        a1To11InQuintile = [0]*5
        a12To18InQuintile = [0]*5
        a19To39InQuintile = [0]*5
        a40To64InQuintile = [0]*5
        over65InQuintile = [0]*5
    
        for agent in self.pop.livingPeople:

            popQuintile[agent.incomeQuintile] += 1
            sizeQuintile[agent.incomeQuintile] += len([x for x in agent.socialContacts.nodes()])
                
            if agent.age >= 1 and agent.age < 12:
                
                a1To11InQuintile[agent.incomeQuintile] += 1
                
                popByAgeCount[0] += 1
                snSizeCount[0] += len([x for x in agent.socialContacts.nodes()])
            if agent.age >= 12 and agent.age < 19:
                
                a12To18InQuintile[agent.incomeQuintile] += 1
                
                popByAgeCount[1] += 1
                snSizeCount[1] += len([x for x in agent.socialContacts.nodes()])
            if agent.age >= 19 and agent.age < 40:
                
                a19To39InQuintile[agent.incomeQuintile] += 1
                
                popByAgeCount[2] += 1
                snSizeCount[2] += len([x for x in agent.socialContacts.nodes()])
            if agent.age >= 40 and agent.age < 65:
                
                a40To64InQuintile[agent.incomeQuintile] += 1
                
                popByAgeCount[3] += 1
                snSizeCount[3] += len([x for x in agent.socialContacts.nodes()])
            if agent.age >= 65:
                
                over65InQuintile[agent.incomeQuintile] += 1
                
                popByAgeCount[4] += 1
                snSizeCount[4] += len([x for x in agent.socialContacts.nodes()])
        averageNetworkSizeByAge = [float(i) / float(j) for i, j in zip(snSizeCount, popByAgeCount)]
        averageNetworkSizeByIncome = [float(i) / float(j) for i, j in zip(sizeQuintile, popQuintile)]
        
#        print '1 to 11 in quintiles: ' + str(a1To11InQuintile)
#        print '12 to 18 in quintiles: ' + str(a12To18InQuintile)
#        print '19 to 39 in quintiles: ' + str(a19To39InQuintile)
#        print '40 to 64 in quintiles: ' + str(a40To64InQuintile)
#        print 'Over 65 in quintiles: ' + str(over65InQuintile)
#        
#        print 'Size network by age group: ' + str(averageNetworkSizeByAge)
#        print 'Size network by income group: ' + str(averageNetworkSizeByIncome)
        
        # pdb.set_trace()
        
    def mobilityRates(self):

        susceptible = [x for x in self.pop.livingPeople if x.healthStatus == 'susceptible']
        shareSusceptible = float(len(susceptible))/float(len(self.pop.livingPeople))

        print 'Deaths to infected ratio: ' + str(self.deathsToInfectedRatio)
        
        risksList = []
        behaviouralIsolationRates = []
        count = 0
        for person in self.pop.livingPeople:
            
            person.contagiousnessIndex = math.pow(person.viralLoad, self.p['loadContagiousnessExp'])
            
            # Determine reduced mobiltiy due to severity of condition
            
            # print 'Share new cases: ' + str(newCasesRatio)
            behaviouralIsolationRate = 1.0
            person.contactReductionRate = 1.0
            if person.hospitalized == True:
                person.contactReductionRate = 0.0
            else:
                person.contactReductionRate = math.pow(1.0 - person.mildConditionIndex, self.p['mildMobilityExp'])
                if self.p['behaviouralResponse'] == True:
                    
                    casesResponseExp = 0
                    behaviouralResponseFactor = self.p['behaviouralResponseFactor']
                    # Reduced mobility due to behavioral response
                    if person.testPositive == False:
                        count += 1
                        workingFactor = 1.0
                        incomeFactor = 1.0
                        if person.status == 'worker':
                            workingFactor = self.p['workingFactorReduction'][person.incomeQuintile]
                            incomeFactor = math.exp(self.p['incomeBehaviourExp']*float(person.incomeQuintile+1))
                        
                        # Risk factor: to substitute to ageFactor
                        risk = self.infectionFatalityRatio[person.ageClass][person.incomeQuintile][0]
                        if person.sex == 'female':
                            risk = self.infectionFatalityRatio[person.ageClass][person.incomeQuintile][1]
                            
                        risksList.append(risk)
                        
                        riskFactor = self.p['riskBehaviourFactor']*risk
                        
                        # ageFactor = math.exp(self.p['ageBehaviourExp']*float(person.ageClass+1))
                        casesResponseExp = behaviouralResponseFactor*incomeFactor*riskFactor*workingFactor # ageFactor
                       
                        behaviouralIsolationRate = 1.0/np.exp(casesResponseExp*self.newCasesIndex)
                        behaviouralIsolationRates.append(behaviouralIsolationRate)
                        
                    elif person.testPositive == True and person.healthStatus == 'infectious':
                        
                        behaviouralResponseFactor *= self.p['socialPreferenceFactor']
                        workingFactor = 1.0
                        incomeFactor = 1.0
                        if person.status == 'worker':
                            workingFactor = self.p['workingFactorReduction'][person.incomeQuintile]
                            incomeFactor = math.exp(self.p['incomeBehaviourExp']*(person.incomeQuintile+1))
                        riskFactor = self.p['riskBehaviourFactor']*self.deathsToInfectedRatio
                        
                        # casesResponseExp = (np.exp(behaviouralResponseExp*(person.incomeQuintile+1)*workingFactor*person.relativeRisk)-1.0)*self.p['scaleFactor']
                        # Alternative formulation
                        casesResponseExp = behaviouralResponseFactor*incomeFactor*riskFactor*workingFactor
                        
                        behaviouralIsolationRate = 1.0/np.exp(casesResponseExp*shareSusceptible)
                        behaviouralIsolationRates.append(behaviouralIsolationRate)
                    
                    if behaviouralIsolationRate < person.contactReductionRate:
                        person.contactReductionRate = behaviouralIsolationRate
        
#        print 'Share of negative test: ' + str(float(count)/float(len(self.pop.livingPeople)))
#        print 'Max risk: ' + str(max(risksList))
#        print 'Mean risk: ' + str(np.mean(risksList))
#        print 'Median risk: ' + str(np.median(risksList))
#        print 'Min risk: ' + str(min(risksList))
#        
#        print 'Max behaviour isolation rate: ' + str(max(behaviouralIsolationRates))
#        print 'Mean behaviour isolation rate: ' + str(np.mean(behaviouralIsolationRates))
#        print 'Median behaviour isolation rate: ' + str(np.median(behaviouralIsolationRates))
#        print 'Min behaviour isolation rate: ' + str(min(behaviouralIsolationRates))
        
        
#        self.minIsolationRate = min(behaviouralIsolationRates)
#        self.maxIsolationRate = max(behaviouralIsolationRates)
#        self.meanIsolationRate = np.mean(behaviouralIsolationRates)
#        self.medianIsolationRate = np.median(behaviouralIsolationRates)
        
        self.maxContacts = max([x.numContacts for x in self.pop.livingPeople])
        
        
        for i in range(5):
            
            reductionRatesInQuintile = [x.contactReductionRate for x in self.pop.livingPeople if x.incomeQuintile == i]
            print 'Quintile ' + str(i) + ' mean reduction rate:' + str(np.mean(reductionRatesInQuintile))
            print 'Quintile ' + str(i) + ' max reduction rate:' + str(max(reductionRatesInQuintile))
            print 'Quintile ' + str(i) + ' min reduction rate:' + str(min(reductionRatesInQuintile))
            print ''
            if i == 0:
                self.mobilityReduction_Q1 = np.mean(reductionRatesInQuintile)
            if i == 1:
                self.mobilityReduction_Q2 = np.mean(reductionRatesInQuintile)
            if i == 2:
                self.mobilityReduction_Q3 = np.mean(reductionRatesInQuintile)
            if i == 3:
                self.mobilityReduction_Q4 = np.mean(reductionRatesInQuintile)
            if i == 4:
                self.mobilityReduction_Q5 = np.mean(reductionRatesInQuintile)
            
        for i in range(int(self.p['ageClasses'])):
            reductionRatesByAge = [x.contactReductionRate for x in self.pop.livingPeople if x.ageClass == i]
            if i == 0:
                self.mobilityReduction_A1 = np.mean(reductionRatesByAge)
            if i == 1:
                self.mobilityReduction_A2 = np.mean(reductionRatesByAge)
            if i == 2:
                self.mobilityReduction_A3 = np.mean(reductionRatesByAge)
            if i == 3:
                self.mobilityReduction_A4 = np.mean(reductionRatesByAge)
            if i == 4:
                self.mobilityReduction_A5 = np.mean(reductionRatesByAge)
            if i == 5:
                self.mobilityReduction_A6 = np.mean(reductionRatesByAge)
            if i == 6:
                self.mobilityReduction_A7 = np.mean(reductionRatesByAge)
            if i == 7:
                self.mobilityReduction_A8 = np.mean(reductionRatesByAge)
            if i == 8:
                self.mobilityReduction_A9 = np.mean(reductionRatesByAge)
            
    
    def domesticInteraction(self):
        
        for person in self.pop.livingPeople:
            
            householdMembers = [x for x in person.house.occupants if x != person]
            # Set of members who are unknowingly infectious
            UIHM = [x for x in householdMembers if x.healthStatus == 'infectious' and x.hospitalized == False and x.testPositive == False]
            # Set of members who are knowingly infectious
            KIHM = [x for x in householdMembers if x.healthStatus == 'infectious' and x.hospitalized == False and x.testPositive == True]
            # person.householdRiskFactor = (float(len(UIHM)) + self.p['householdIsolationFactor']*float(len(KIHM)))/float(len(householdMembers))
            uihmViralLoad = 0
            if len(UIHM) > 0:
                uihmViralLoad = sum([x.contagiousnessIndex for x in UIHM])
            kihmViralLoad = 0
            if len(KIHM):
                kihmViralLoad = sum([x.contagiousnessIndex for x in KIHM])*self.p['householdIsolationFactor']
            if len(householdMembers) > 1:
                # The domestic risk increases with the number of infected members, but less than linearly ('householdInfectionExp' parameter less than 1)
                person.domesticRiskFactor = (uihmViralLoad + kihmViralLoad)/np.power(float(len(householdMembers)), self.p['householdInfectionExp'])
            else:
                person.domesticRiskFactor = 0
    
    
        
    def createInfoNetworks(self):
        
        for house in self.map.occupiedHouses:
            house.beta = self.p['initialBeta']
            
        # Create kinship network
        self.kinshipNetwork()
        
        # Assign income quintiles to agents
        households = [x for x in self.map.occupiedHouses]
        # households.sort(key=operator.attrgetter("incomePerCapita"))
        households.sort(key=operator.attrgetter("householdIncome"))
        for i in range(int(self.p['incomeClasses'])):
            number = int(round(len(households)/(self.p['incomeClasses']-float(i))))
            quintile = households[:number]
            for j in quintile:
                j.incomeQuintile = i
                for agent in j.occupants:
                    agent.incomeQuintile = i
            households = [x for x in households if x not in quintile]
        
        for person in self.pop.livingPeople:
            person.socialContacts.clear()
        
        # Create a 9-age-groups contacts matrix ############################################################################
        ageShares = []
        numClasses = int(self.contacts.shape[1])
        if (numClasses%2) != 0:
            numClasses -= 1
        for i in range(0, numClasses, 2):
            ageClass_1 = len([x for x in self.pop.livingPeople if x.interactionAgeClass == i])
            ageClass_2 = len([x for x in self.pop.livingPeople if x.interactionAgeClass == i+1])
            share_1 = float(ageClass_1)/float(ageClass_1+ageClass_2)
            share_2 = float(ageClass_2)/float(ageClass_1+ageClass_2)
            shares = [share_1, share_2]
            ageShares.extend(shares)
            
        # 9-classes contacts matrix
        newMatrix = []
        for i in range(int(self.contacts.shape[1])):
            newColumn = []
            col = list(self.contacts[i])
            numCol = int(len(col))
            if (numCol%2) != 0:
                numCol -= 1
            for j in xrange(0, numCol, 2):
                newColumn.append(col[j]+col[j+1])
            if (len(col)%2) != 0:
                newColumn.append(col[-1])
            newMatrix.append(newColumn)
        self.contactsMatrix = []
        numCol = int(len(newMatrix))
        if (numCol%2) != 0:
            numCol -= 1
        for i in range(0, numCol, 2):
            self.contactsMatrix.append([(a*ageShares[i] + b*ageShares[i+1]) for a, b in zip(newMatrix[i], newMatrix[i+1])])
        if (len(newMatrix)%2) != 0:
            self.contactsMatrix.append(newMatrix[-1])
        
        ############################################################################################################################
        
        maxContacts = []
        contactBias = self.p['classContactBias']
        if self.p['5yearAgeClasses'] == False:
            ageClasses = int(self.p['ageClasses'])
            contactsByAge = self.contactsMatrix
        else:
            ageClasses = int(self.p['interactionAgeClasses'])
            contactsByAge = self.contacts
            
        for i in range(ageClasses):
            if i <= 12:
                for j in range(ageClasses):
                    if j > 12:
                        contactsByAge[i][j] *= self.p['Over65ContactIncrement']
            else:
                for j in range(ageClasses):
                    contactsByAge[i][j] *= self.p['Over65ContactIncrement']
            
        self.classContactsMatrix = []
        # Introducing contact increment to increase contacts in older people ##
        contactIncrement = self.p['contactCompoundFactor']
        for i in range(ageClasses): # 
            contactsByClass = []
            ageGroupContacts = []
            # For each age group, compute the total contacts with all the other age groups (incremented by a factor)
            # It determines the size of the ego network
           
            totalContacts = sum(contactsByAge[i])*contactIncrement
            # Given these total contacts for the class has a whole, find the contacts fro this age group for each income quintile class
            a = 0
            for z in range(int(self.p['incomeClasses'])):
                a += (1.0/float(self.p['incomeClasses']))*math.pow(contactBias, z)
            lowClassContacts = totalContacts/a
            oldMax = 0
            for j in range(int(self.p['incomeClasses'])):
                classContacts = float(lowClassContacts*math.pow(contactBias, j))
                
                print 'Class ' + str(j) + ' contacts: ' + str(classContacts)
                
                
                newMax = float(max(np.random.poisson(math.ceil(classContacts), 1000)))
                while newMax < oldMax:
                    newMax = float(max(np.random.poisson(math.ceil(classContacts), 1000)))
                # For each income quintile, the max size of social network
                contactsByClass.append(newMax)
                oldMax = newMax
                # contactsByClass.append(np.ceil(classContacts))
              
            maxContacts.append(contactsByClass)
            
            # pdb.set_trace()
            
            # An age-age contact matrix is created for each income quintile.
            for j in range(ageClasses):
                classContacts = []
                lowClassContacts = float(contactsByAge[i][j])/a
                
                for z in range(int(self.p['incomeClasses'])):
                    classContacts.append(lowClassContacts*math.pow(contactBias, z)*contactIncrement)
                
                
                ageGroupContacts.append(classContacts)
            
            self.classContactsMatrix.append(ageGroupContacts)
     
        print 'Max Contacts: ' + str(maxContacts)
        
        for person in self.pop.livingPeople:
            person.socialContacts.add_node(person)
        
        if self.p['5yearAgeClasses'] == False:
            ageClasses = int(self.p['ageClasses'])
        else:
            ageClasses = int(self.p['interactionAgeClasses'])
        
        self.totalAge1Age2Income1Contacts = []
        for i in range(ageClasses):
            totalAge2Contacts = []
            for j in range(int(self.p['incomeClasses'])):
                groupMaxContacts = maxContacts[i][j] # Total size of network per age and income group
                
                print 'Max contacts for age ' + str(i) + ' and income class ' + str(j) + ': ' + str(groupMaxContacts)
                
                totContacts = sum([self.classContactsMatrix[i][x][j] for x in range(ageClasses)])
                
                
                print 'Tot contacts: ' + str(totContacts)
                
                if self.p['5yearAgeClasses'] == False:
                    socialGroup = [x for x in self.pop.livingPeople if x.ageClass == i and x.incomeQuintile == j]
                else:
                    socialGroup = [x for x in self.pop.livingPeople if x.interactionAgeClass == i and x.incomeQuintile == j]
                totalIncome1Contacts = []
                for z in range(ageClasses):
                    if self.p['5yearAgeClasses'] == False:
                        contactsAgeGroup = [x for x in self.pop.livingPeople if x.ageClass == z]
                    else:
                        contactsAgeGroup = [x for x in self.pop.livingPeople if x.interactionAgeClass == z]
                    
                    # contactsMeanRate = np.mean([x.contactReductionRate for x in contactsAgeGroup])
                    # 1 - Compute the daily contacts of agent in age i and class j with agents of age z
                    # 2 - Compute total contacts of class
                    # 3 - Start a cicle to assign contacts to members of social group [i, j]
                    shareContacts = self.classContactsMatrix[i][z][j]/float(totContacts)
                    sizeGroup = len(socialGroup)
                    print 'Share of contacts with age ' + str(z) + ' by agent of age class ' + str(i) + ' and income class ' + str(j) + ': ' + str(shareContacts)
                    print 'Size of social group: ' + str(sizeGroup)
                    print 'Size of network in this group: ' + str(groupMaxContacts)
                    
                    totalAgeContacts = int(round((shareContacts)*float(groupMaxContacts*sizeGroup)))
                    
                    print 'Total contacts: ' + str(totalAgeContacts)
                    
                    totalIncome1Contacts.append(totalAgeContacts)
                totalAge2Contacts.append(totalIncome1Contacts)
            self.totalAge1Age2Income1Contacts.append(totalAge2Contacts)
            # age1Age2Income1Contacts.append(totalAge2Contacts)
        # pdb.set_trace()
        
        age1Age2Income1Contacts = copy.deepcopy(self.totalAge1Age2Income1Contacts)
        # Find totalcontacts for each age-age-income combination: totalAgeContacts = [[[]]]
        for i in range(ageClasses):
            for j in range(int(self.p['incomeClasses'])):
                groupMaxContacts = maxContacts[i][j]
                if self.p['5yearAgeClasses'] == False:
                    socialGroup = [x for x in self.pop.livingPeople if x.ageClass == i and x.incomeQuintile == j]
                else:
                    socialGroup = [x for x in self.pop.livingPeople if x.interactionAgeClass == i and x.incomeQuintile == j]
                for z in range(ageClasses):
                    
                    print 'Doing age class ' + str(i) + ' of income class ' + str(j) + ' with age class ' + str(z)
                    print 'Num contacts: ' + str(age1Age2Income1Contacts[i][j][z])
                    
                    if self.p['5yearAgeClasses'] == False:
                        contactsAgeGroup = [x for x in self.pop.livingPeople if x.ageClass == z]
                    else:
                        contactsAgeGroup = [x for x in self.pop.livingPeople if x.interactionAgeClass == z]
                
                    remainingAgents = [x for x in socialGroup if x.socialContacts.number_of_nodes() < groupMaxContacts]
                    while age1Age2Income1Contacts[i][j][z] > 0 and len(remainingAgents) > 0:
                        agent = np.random.choice(remainingAgents)
                        contactsGroup = [x for x in contactsAgeGroup if x not in agent.house.occupants and x not in agent.socialContacts.nodes() and x.socialContacts.number_of_nodes() < maxContacts[z][x.incomeQuintile] and age1Age2Income1Contacts[z][x.incomeQuintile][i] > 0]
                        # The probability of becoming part of an agent's contacts list depends on social class, location distances and mobility.
                        if agent.socialContacts.number_of_nodes() < groupMaxContacts and len(contactsGroup) > 0:
                            friends = [x for x in agent.socialContacts.nodes()]
                            commonFriends = [len([y for y in x.socialContacts.nodes() if y in friends]) for x in contactsGroup]
                            friendsExp = [self.p['friendsInteractionBeta']*math.pow(float(x), self.p['friendsInteractionExp']) for x in commonFriends]
                            classDistances = [abs(x.incomeQuintile-agent.incomeQuintile) for x in contactsGroup]
                            geoDistances = [self.geoDistance(agent.house, x.house) for x in contactsGroup]
                            classExps = [self.p['classInteractionBeta']*math.pow(x, self.p['classInteractionExp']) for x in classDistances]
                            distanceExps = [self.p['locationInteractionBeta']*math.pow(x, self.p['locationInteractionExp']) for x in geoDistances]
                            totalExps = [(c+d-f) for c, d, f in zip(classExps, distanceExps, friendsExp)]
                            weightsFullMobility = [1.0/math.exp(x) for x in totalExps]
                            # interactionRates = [x.contactReductionRate for x in contactsGroup]
                            # weights = [w*r for w, r in zip(weightsFullMobility, interactionRates)]
                            contactsProbs = [x/sum(weightsFullMobility) for x in weightsFullMobility]
                            potentialContact = np.random.choice(contactsGroup, p = contactsProbs)
                            quintile = potentialContact.incomeQuintile
                            contactIndex = contactsGroup.index(potentialContact)
                            contactWeight = weightsFullMobility[contactIndex]
                            agent.socialContacts.add_edge(agent, potentialContact, weight = contactWeight)
                            potentialContact.socialContacts.add_edge(potentialContact, agent, weight = contactWeight)
                            age1Age2Income1Contacts[i][j][z] -= 1
                            age1Age2Income1Contacts[z][quintile][i] -= 1
                            if potentialContact.house not in agent.house.infoNetwork.nodes():
                                contactWeight = 1.0/(1.0+distanceExps[contactIndex])
                                agent.house.infoNetwork.add_edge(agent.house, potentialContact.house, weight = contactWeight)
                                potentialContact.house.infoNetwork.add_edge(potentialContact.house, agent.house, weight = contactWeight)
                        else:
                            remainingAgents.remove(agent)
        
        # Add random connections....
        # Assign to each person a number of small-world (SW) connections, through a Poisson draw with mean proportional to income quintile
        # For each agent with SW connections > 0, sample the number of agents from general population (with age boundaries)
        # In the social contact function, each agent in the swContacts list is met with a certain probability.
        workingAdults = [x for x in self.pop.livingPeople if x.age > 17 and x.age < 65]
        for agent in workingAdults:
            mean_SWcontacts = float(agent.socialContacts.number_of_nodes())*self.p['fractionSWcontacts']
            swContacts = np.random.poisson(mean_SWcontacts)
            agent.totalContacts = len([x for x in agent.socialContacts.nodes()])+swContacts
        
        for agent in workingAdults:
            if agent.socialContacts.number_of_nodes() < agent.totalContacts:
                otherTownsPeople = [x for x in workingAdults if x.house.town != agent.house.town]
                contactsGroup = [x for x in otherTownsPeople if x not in agent.house.occupants and x not in agent.socialContacts.nodes() and x.socialContacts.number_of_nodes() < x.totalContacts]
                if len(contactsGroup) > 0:
                    additionalContacts = agent.totalContacts-len(agent.dailyContacts)
                    if additionalContacts > len(contactsGroup):
                        additionalContacts = len(contactsGroup)
                    classDistances = [abs(x.incomeQuintile-agent.incomeQuintile) for x in contactsGroup]
                    geoDistances = [self.townDistance(agent.house.town, x.house.town) for x in contactsGroup]
                    classExps = [self.p['classInteractionBeta']*math.pow(x, self.p['classInteractionExp']) for x in classDistances]
                    distanceExps = [self.p['townInteractionBeta']*math.pow(x, self.p['townInteractionExp']) for x in geoDistances]
                    totalExps = [(c+d) for c, d, in zip(classExps, distanceExps)]
                    weightsFullMobility = [1.0/math.exp(x) for x in totalExps]
                    # interactionRates = [x.contactReductionRate for x in contactsGroup]
                    # weights = [w*r for w, r in zip(weightsFullMobility, interactionRates)]
                    contactsProbs = [x/sum(weightsFullMobility) for x in weightsFullMobility]
                    swContacts = np.random.choice(contactsGroup, additionalContacts, replace = False, p = contactsProbs)
                    contactWeights = [weightsFullMobility[contactsGroup.index(x)] for x in swContacts]
                    for i in range(len(swContacts)):
                        agent.socialContacts.add_edge(agent, swContacts[i], weight = contactWeights[i])
                        swContacts[i].socialContacts.add_edge(swContacts[i], agent, weight = contactWeights[i])
                
        
#        for k in range(len(self.pop.livingPeople)):
#            randomAgent = np.random.choice(self.pop.livingPeople)
#            randomContact = np.random.choice([x for x in self.pop.livingPeople if x not in randomAgent.socialContacts.nodes()])
#            randomFriend = np.random.choice([x for x in randomAgent.socialContacts.nodes() if x != randomAgent])
#            # Remove old link
#            randomAgent.socialContacts.remove_node(randomFriend)
#            randomFriend.socialContacts.remove_node(randomAgent)
#            
#            friends = [x for x in randomAgent.socialContacts.nodes()]
#            commonFriends = len([x for x in randomContact.socialContacts.nodes() if x in friends])
#            friendsExp = self.p['friendsInteractionBeta']*math.pow(float(commonFriends), self.p['friendsInteractionExp'])
#            classDistance = abs(randomContact.incomeQuintile-randomAgent.incomeQuintile)
#            geoDistance = self.geoDistance(randomAgent.house, randomContact.house)
#            classExps = self.p['classInteractionBeta']*math.pow(classDistance, self.p['classInteractionExp'])
#            distanceExps = self.p['locationInteractionBeta']*math.pow(geoDistance, self.p['locationInteractionExp'])
#            weightLink = 1.0/math.exp(classExps+distanceExps-friendsExp)
#            # Create new link
#            randomAgent.socialContacts.add_edge(randomAgent, randomContact, weight = weightLink)
#            randomContact.socialContacts.add_edge(randomContact, randomAgent, weight = weightLink)
        
    
#    def socialInteraction_Debug(self):
#        # In this function, the agent draws from his networks a number of daily contacts which depends on his isolation behaviour.
#        # The probability that a node is selected as daily contact depends on the links' weights,
#        
#        ## print 'Class Contacts Matrix: ' + str(self.classContactsMatrix)
#        
#        susceptible = [x for x in self.pop.livingPeople if x.healthStatus == 'susceptible']
#        for agent in susceptible:
#            agent.dailyContacts = []
#            # 1 - Get daily contacts
#            friends = [x for x in agent.socialContacts.nodes() if x != agent and x.dead == False and x.hospitalized == False and (agent.socialContacts[agent][x]['weight']*x.contactReductionRate) > 0]
#            if len(friends) > 0:
#                quintile = agent.incomeQuintile
#                if self.p['5yearAgeClasses'] == False:
#                    ageClasses = int(self.p['ageClasses'])
#                    agentAgeGroup = agent.ageClass
#                else:
#                    ageClasses = int(self.p['interactionAgeClasses'])
#                    agentAgeGroup = agent.interactionAgeClass
#                averageContacts = sum([self.classContactsMatrix[agentAgeGroup][x][quintile] for x in range(ageClasses)])*self.p['dailyContactsShare']
#                if self.lockdown == True:
#                    averageContacts *= self.p['lockdownContactReductionRate']
#                
#                den = sum([agent.socialContacts[agent][x]['weight'] for x in friends])
#                num = sum([agent.socialContacts[agent][x]['weight']*x.contactReductionRate for x in friends])
#                averageIsolationRate = 0
#                if den > 0:
#                    averageIsolationRate = num/den
#                meanNumContacts = float(averageContacts)*agent.contactReductionRate*averageIsolationRate
#                agent.numContacts = np.random.poisson(meanNumContacts)
#                # Now, the contacts are randomly sampled from the agent's social network
#                if agent.numContacts > len(friends):
#                    agent.numContacts = len(friends)
#        
#        # Random contacts
#        totContacts = []
#        for agent in susceptible:
#            friends = [x for x in self.pop.livingPeople if x != agent and x.dead == False and x.hospitalized == False and x.contactReductionRate > 0]
#            agent.dailyContacts = np.random.choice(friends, agent.numContacts, replace = False)
#            
#            totContacts.append(agent.numContacts)
#        
#        print 'Average num of contacts: ' + str(np.mean(totContacts))
    
    def commutingProcess(self):
        commutingAgents = [x for x in self.pop.livingPeople if x.age > 17 and x.age < 65]
        for agent in commutingAgents:
            # With a certain probability (parameter) an agent visits another town
            if np.random.random() < self.p['commutingProb']:
                towns = [x for x in self.map.towns if x != agent.house.town]
                pop = [len(x.people) for x in towns]
                dist = [self.manhattanDistance(agent.house.town, x)*0.05 for x in towns]
                # The probability of the commuting town depends on size of town and distance from agent's town
                weights = [float(x)/np.power(float(y), 2.0) for x, y in zip(pop, dist)]
                probs = [x/sum(weights) for x in weights]
                agent.commutingTown = np.random.choice(towns, p = probs)
            else:
                agent.commutingTown = None
                

    def socialInteraction_SPE_II(self):
        
        # In this function, the agents are assigned to venues, in a way consistent with the empirical contact matrix.
        # First, the venues are emptied...
        
        # Then, agents are randomly allocated to venues.
        # np.random.shuffle(self.pop.livingPeople)
        
        if self.p['5yearAgeClasses'] == False:
            ageClasses = int(self.p['ageClasses'])
        else:
            ageClasses = int(self.p['interactionAgeClasses'])
      

        remainingContacts = self.totalInteractions
        
                    # currentDistance += np.power(float(abs(self.totalAge1Age2Income1Contacts[i][j][z]-age1Age2Income1Contacts[i][j][z])), self.p['contactsDistanceExp'])
        for contactClass in self.classesContacts:
            contactClass.actualContactsByAge = [0]*int(self.p['interactionAgeClasses'])

        effectiveContacts = []
        for i in range(int(self.p['interactionAgeClasses'])):
            incomeContacts = []
            for j in range(int(self.p['incomeClasses'])):
                incomeContacts.append(0)
            effectiveContacts.append(incomeContacts)
       
        self.newVenues = 0
        
        while remainingContacts > 0:
            # 1 - A venue is randomly chosen
            venue = np.random.choice(self.map.allVenues)
            # 2 - All the agents in the venue's town (who did not commute to anotehr town) are selected
            agents = [x for x in self.pop.livingPeople if x.house.town == venue.town and x.commutingTown == None]
            # The agents commuting to the venue's town are added
            agents.extend([x for x in self.pop.livingPeople if x.commutingTown == venue.town]) # Add commuters
            emptyClasses = []
            for i in range(int(self.p['interactionAgeClasses'])):
                for j in range(int(self.p['incomeClasses'])):
                    # Select age/income interaction info
                    intClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
                    # Select all the agents of that age/income class
                    potentialVisitors = [x for x in agents if x.interactionAgeClass == intClass.age and x.incomeQuintile == intClass.quintile]
                    # There is no one in that age/class group
                    if len(potentialVisitors) == 0:
                        emptyClasses.append(intClass)
                        
            # Loop until find a venue in a town with at least one age/class groups with people
            while len(emptyClasses) == len(self.classesContacts):
                venue = np.random.choice(self.map.allVenues)
                agents = [x for x in self.pop.livingPeople if x.house.town == venue.town and x.commutingTown == None]
                agents.extend([x for x in self.pop.livingPeople if x.commutingTown == venue.town]) # Add commuters
                emptyClasses = []
                for i in range(int(self.p['interactionAgeClasses'])):
                    for j in range(int(self.p['incomeClasses'])):
                        intClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
                        potentialVisitors = [x for x in agents if x.interactionAgeClass == intClass.age and x.incomeQuintile == intClass.quintile]
                        if len(potentialVisitors) == 0:
                            emptyClasses.append(intClass)
            
            # Select an age/income group based on number of residual contacts 
            # Two separate processes are run: one if the venue is empty.
            # One if the venue is occupaied by someone
            
            if len(venue.occupants) == 0: 
                for i in range(int(self.p['interactionAgeClasses'])):
                    for j in range(int(self.p['incomeClasses'])):
                        ageIncomeClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
                        # The probabliity of an age/income class being selected depends on the residual number of contacts for that class
                        weightsClass = ageIncomeClass.contacts-effectiveContacts[i][j]
                        ageIncomeClass.weight = float(np.exp(self.p['deltaContactsBeta']*weightsClass))
                weightsClasses = [x.weight for x in self.classesContacts]
                potentialVisitors = []
                visitedClasses = []
                # A loop is strated where an age/income class is selected and the list of people in that call is created
                while len(potentialVisitors) == 0:
                    remainingClasses = [x for x in self.classesContacts if x not in visitedClasses]
                    weightsClasses = [x.weight for x in remainingClasses]
                    probsClasses = [float(x)/sum(weightsClasses) for x in weightsClasses]
                    intClass = np.random.choice(remainingClasses, p = probsClasses)
                    if intClass not in visitedClasses:
                        visitedClasses.append(intClass)
                    potentialVisitors = [x for x in agents if x.interactionAgeClass == intClass.age and x.incomeQuintile == intClass.quintile]
                
                # Create list of distances of people in the selected age/income group from the venue
                distances = [self.manhattanDistance(x.house, venue) for x in potentialVisitors]
                
                meanDist = np.mean(distances)
                if meanDist <= 0:
                    meanDist = 1.0
                newDist = []
                for x in distances:
                    if x == 0:
                        newDist.append(meanDist)
                    else:
                        newDist.append(x)
                distances = [x for x in newDist]
                
                weights = [1.0/np.power(x, self.p['distanceBetaVenue']) for x in distances]
                probs = [x/sum(weights) for x in weights]
#                if sum(probs) < 1.0:
#                    print ''
#                    print 'Error: probs do not add to !'
#                    print weights
#                    print probs
#                    print len(potentialVisitors)
#                    print 'Venue coordinates: ' + str(venue.x) + ' - ' + str(venue.y)
#                    print ''
                
                # An agent is selected and added to the venue
                agent = np.random.choice(potentialVisitors, p = probs)
                venue.occupants.append(agent)
                venue.numVirtualOccupants += 1
               
            else:
                # In this case, the probability of an age/income class being selected depends on the venue's attendants
                weightsAges = []
                for j in range(int(self.p['interactionAgeClasses'])):
                    deltaContacts = []
                    for agent in venue.occupants:
                        i = agent.interactionAgeClass
                        z = agent.incomeQuintile
                        # Retrieve the age/income class for each agent in the venue
                        intClass = [x for x in self.classesContacts if x.age == i and x.quintile == z][0]
                        # Compute the residual contacts for that age/income group for that age
                        deltaAgeContacts = float(intClass.totContactsByAge[j]-intClass.actualContactsByAge[j])
                        deltaContacts.append(deltaAgeContacts)
                    # Average of residual contacts for all the age/income classes of the agents in the venue
                    meanDeltaContacts = np.mean(deltaContacts)
                    weightsAges.append(float(np.exp(self.p['deltaContactsBeta']*meanDeltaContacts)))
                    
                if sum(weightsAges) > 0:
                    probs = [float(x)/sum(weightsAges) for x in weightsAges]
                    # Select the 
                    ageClass = np.random.choice(range(int(self.p['interactionAgeClasses'])), p = probs)
                    potentialVisitors = [x for x in agents if x.interactionAgeClass == ageClass and x not in venue.occupants]
                    
                elif sum(weightsAges) == 0 or len(potentialVisitors) == 0:
                    if sum(weightsAges) == 0:
                        print 'Weights age equal zero'
                    if len(potentialVisitors) == 0:
                        print 'Len visitors equal zero'
                    visitedAges = []
                    cond = False
                    while cond == False:
                        ageClass = np.random.choice(range(int(self.p['interactionAgeClasses'])))
                        if ageClass not in visitedAges:
                            visitedAges.append(ageClass)
                        potentialVisitors = [x for x in agents if x.interactionAgeClass == ageClass and x not in venue.occupants]
                        if len(potentialVisitors) > 0:
                            cond = True
                        if len(potentialVisitors) == 0 and len(visitedAges) == int(self.p['interactionAgeClasses']):
                            print 'Error: all the age classes have zero potential visitors!'
                            sys.exit()
                
                if len(potentialVisitors) > 0:
                    
                    geoDistances = [self.manhattanDistance(x.house, venue) for x in potentialVisitors]
                    meanVenueIQ = np.mean([float(x.incomeQuintile) for x in venue.occupants])
                    classDistances = [(x.incomeQuintile-meanVenueIQ) for x in potentialVisitors]
                    
                    meanDist = np.mean(geoDistances)
                    if meanDist <= 0:
                        meanDist = 1.0
                    newDist = []
                    for x in geoDistances:
                        if x == 0:
                            newDist.append(meanDist)
                        else:
                            newDist.append(x)
                    geoDistances = [x for x in newDist]
                    
                    classExps = [self.p['classInteractionBeta']*math.pow(x, self.p['classInteractionExp']) for x in classDistances]
                    
                    distanceExps = [self.p['locationInteractionBeta']*math.pow(x, self.p['locationInteractionExp']) for x in geoDistances]
                    
                    totalDistances = [(h+c) for h, c in zip(classExps, distanceExps)]
                    
                    weights = [1.0/math.exp(x*self.p['venueDistanceExp']) for x in totalDistances]
                    
                    probs = [x/sum(weights) for x in weights]
                    
                    # The new venue's attendant is selected
                    agent = np.random.choice(potentialVisitors, p = probs)
                    
                    
                    
#                    if agent.hospitalized == False:
#                        if self.p['behaviourModuleOn'] == True:
#                            probAttendance = self.computeAttendanceProb(agent, venue)
#                            if np.random.random() < probAttendance:
#                                venue.occupants.append(agent)
#                            else:
#                                self.cumulatedAbsences += 1
#                        else:
#                            venue.occupants.append(agent)
                            
                    venue.occupants.append(agent)   
                    
                    numContacts = int(math.ceil(math.pow(len(venue.occupants)-1, self.p['venueBetaExp'])))
                    
                    lastRemCont = remainingContacts
                    # The new agent increases the number of interactions proportionally to the venue's population, but less than linearly
                    remainingContacts -= 2*numContacts
                    venue.numVirtualOccupants += 1
                    
                    if remainingContacts == lastRemCont:
                        print 'Error: rem contacts do not diminish!'
                        print remainingContacts
                        print venue.numVirtualOccupants
                        print len(potentialVisitors)
                        sys.exit()
                    
                    otherAttendants = [x for x in venue.occupants if x != agent]
                    # Select agent's contacts
                    newContacts = np.random.choice(otherAttendants, numContacts, replace = False)
                    # Update age/incoem groups' contacts
                    i = agent.interactionAgeClass
                    j = agent.incomeQuintile
                    agentClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
                    effectiveContacts[i][j] += numContacts
                    for attendant in newContacts:
                        agentClass.actualContactsByAge[attendant.interactionAgeClass] += 1
                        i = attendant.interactionAgeClass
                        j = attendant.incomeQuintile
                        attendantClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
                        attendantClass.actualContactsByAge[agent.interactionAgeClass] += 1
                        effectiveContacts[i][j] += 1
    
        for venue in self.map.allVenues:
            # Now, the actual list of occupants is determiend by removing the agents according to their propensity to isolate
            isolatedAgents = []
            for agent in venue.occupants:
                if np.random.random() < agent.isolationRate:
                    isolatedAgents.append(agent)
                    
            venue.occupants = [x for x in venue.occupants if x not in isolatedAgents]
            self.cumulatedAbsences += len(isolatedAgents)
#            actualVisitors = []
#            for agent in venue.occupants:
#                probAttendance = self.computeAttendanceProb(agent, len(venue.occupants))
#                if np.random.random() < agent.contactReductionRate:
#                    actualVisitors.append(agent)
#            venue.occupants = [x for x in actualVisitors]
            venue.pastAttendants.append(len(venue.occupants))
            # Update past attendance
            
            attendancesDiscounted = 0
            den = 0
            for i in range(len(venue.pastAttendants)):
                index = len(venue.pastAttendants)-1
                attendancesDiscounted += venue.pastAttendants[index-i]*np.power(self.p['timeDiscountingFactor'], i)
                den += np.power(self.p['timeDiscountingFactor'], i)
            venue.pastAttendance = float(attendancesDiscounted)/float(den)


                
#    def socialInteraction_SPE(self):
#        
#        # In this function, the agents are assigned to venues, in a way consistent with the empirical contact matrix.
#        # First, the venues are emptied...
#        
#        # Then, agents are randomly allocated to venues.
#        # np.random.shuffle(self.pop.livingPeople)
#        
#        if self.p['5yearAgeClasses'] == False:
#            ageClasses = int(self.p['ageClasses'])
#        else:
#            ageClasses = int(self.p['interactionAgeClasses'])
#      
#
#        remainingContacts = self.totalInteractions
#        
#                    # currentDistance += np.power(float(abs(self.totalAge1Age2Income1Contacts[i][j][z]-age1Age2Income1Contacts[i][j][z])), self.p['contactsDistanceExp'])
#        totalFitness = 0
#        
#        for contactClass in self.classesContacts:
#            contactClass.actualContactsByAge = [0]*int(self.p['interactionAgeClasses'])
#
#        effectiveContacts = []
#        for i in range(int(self.p['interactionAgeClasses'])):
#            incomeContacts = []
#            for j in range(int(self.p['incomeClasses'])):
#                incomeContacts.append(0)
#            effectiveContacts.append(incomeContacts)
#       
#        self.newVenues = 0
#        
#        while remainingContacts > 0:
#        
#            # At each cicle, create a random venue with a certain prob.
##            if np.random.random() < self.p['probNewVenue']:
##                # Selecet randomly a town
##                popTowns = [len(x.people) for x in self.map.towns]
##                probs = [float(x)/sum(popTowns) for x in popTowns]
##                town = np.random.choice(self.map.towns, p = probs)
##                emptyVenues = [x for x in town.venues if len(x.occupants) == 0]
##                if len(emptyVenues) == 0:
##                    occupiedSpots = [[i.x, i.y] for i in town.houses]
##                    occupiedSpots.extend([[i.x, i.y] for i in town.venues])
##                    newVenue = Venue(town)
##                    # Determine venue's location
##                    randomX = random.randint(0, int(self.p['townGridDimension']))
##                    randomY = random.randint(0, int(self.p['townGridDimension']))
##                    while [randomX, randomY] in occupiedSpots:
##                        'Print occupied spot.'
##                        randomX = random.randint(0, int(self.p['townGridDimension']))
##                        randomY = random.randint(0, int(self.p['townGridDimension']))
##                        
##                    newVenue.x = randomX
##                    newVenue.y = randomY
##                    town.venues.append(newVenue)
##                    self.map.allVenues.append(newVenue)
##                    self.newVenues += 1
##                    print 'New venue created.'
#            
#            # 1 - A venue is randomly chosen
#            venue = np.random.choice(self.map.allVenues)
#            # 2 - All the agents in the venue's town (who did not commute to anotehr town) are selected
#            agents = [x for x in self.pop.livingPeople if x.house.town == venue.town and x.commutingTown == None]
#            # The agents commuting to the venue's town are added
#            agents.extend([x for x in self.pop.livingPeople if x.commutingTown == venue.town]) # Add commuters
#            emptyClasses = []
#            for i in range(int(self.p['interactionAgeClasses'])):
#                for j in range(int(self.p['incomeClasses'])):
#                    # Select age/income interaction info
#                    intClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
#                    # Select all the agents of that age/income class
#                    potentialVisitors = [x for x in agents if x.interactionAgeClass == intClass.age and x.incomeQuintile == intClass.quintile]
#                    if len(potentialVisitors) == 0:
#                        emptyClasses.append(intClass)
#            while len(emptyClasses) == len(self.classesContacts):
#                venue = np.random.choice(self.map.allVenues)
#                agents = [x for x in self.pop.livingPeople if x.house.town == venue.town and x.commutingTown == None]
#                agents.extend([x for x in self.pop.livingPeople if x.commutingTown == venue.town]) # Add commuters
#                emptyClasses = []
#                for i in range(int(self.p['interactionAgeClasses'])):
#                    for j in range(int(self.p['incomeClasses'])):
#                        intClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
#                        potentialVisitors = [x for x in agents if x.interactionAgeClass == intClass.age and x.incomeQuintile == intClass.quintile]
#                        if len(potentialVisitors) == 0:
#                            emptyClasses.append(intClass)
#                    
#            # Select an age/income group based on number of residual contacts 
#            if len(venue.occupants) == 0: 
#               
#                for i in range(int(self.p['interactionAgeClasses'])):
#                    for j in range(int(self.p['incomeClasses'])):
#                        ageIncomeClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
#                        # The probabliity of an age/income class being selected depends on the residual number of contacts for that class
#                        weightsClass = ageIncomeClass.contacts-effectiveContacts[i][j]
#                        ageIncomeClass.weight = float(np.exp(self.p['deltaContactsBeta']*weightsClass))
#                weightsClasses = [x.weight for x in self.classesContacts]
#                potentialVisitors = []
#                visitedClasses = []
#                while len(potentialVisitors) == 0:
#                    remainingClasses = [x for x in self.classesContacts if x not in visitedClasses]
#                    weightsClasses = [x.weight for x in remainingClasses]
#                    probsClasses = [float(x)/sum(weightsClasses) for x in weightsClasses]
#                    intClass = np.random.choice(remainingClasses, p = probsClasses)
#                    if intClass not in visitedClasses:
#                        visitedClasses.append(intClass)
#                    potentialVisitors = [x for x in agents if x.interactionAgeClass == intClass.age and x.incomeQuintile == intClass.quintile]
#                distances = [self.manhattanDistance(x.house, venue) for x in potentialVisitors]
#                
#                meanDist = np.mean(distances)
#                if meanDist <= 0:
#                    meanDist = 1.0
#                newDist = []
#                for x in distances:
#                    if x == 0:
#                        newDist.append(meanDist)
#                    else:
#                        newDist.append(x)
#                distances = [x for x in newDist]
#                
#                weights = [1.0/np.power(x, self.p['distanceBetaVenue']) for x in distances]
#                probs = [x/sum(weights) for x in weights]
##                if sum(probs) < 1.0:
##                    print ''
##                    print 'Error: probs do not add to !'
##                    print weights
##                    print probs
##                    print len(potentialVisitors)
##                    print 'Venue coordinates: ' + str(venue.x) + ' - ' + str(venue.y)
##                    print ''
#                agent = np.random.choice(potentialVisitors, p = probs)
#                venue.occupants.append(agent)
#                venue.numVirtualOccupants += 1
#               
#            else:
#                # In this case, the probability of an age/income class being selected depends on the venue's attendants
#                weightsAges = []
#                for j in range(int(self.p['interactionAgeClasses'])):
#                    deltaContacts = []
#                    for agent in venue.occupants:
#                        i = agent.interactionAgeClass
#                        z = agent.incomeQuintile
#                        intClass = [x for x in self.classesContacts if x.age == i and x.quintile == z][0]
#                        deltaAgeContacts = float(intClass.totContactsByAge[j]-intClass.actualContactsByAge[j])
#                        deltaContacts.append(deltaAgeContacts)
#                    meanDeltaContacts = np.mean(deltaContacts)
#                    weightsAges.append(float(np.exp(self.p['deltaContactsBeta']*meanDeltaContacts)))
#                if sum(weightsAges) > 0:
#                    probs = [float(x)/sum(weightsAges) for x in weightsAges]
#                    ageClass = np.random.choice(range(int(self.p['interactionAgeClasses'])), p = probs)
#                    potentialVisitors = [x for x in agents if x.interactionAgeClass == ageClass and x not in venue.occupants]
#                elif sum(weightsAges) == 0 or len(potentialVisitors) == 0:
#                    if sum(weightsAges) == 0:
#                        print 'Weights age equal zero'
#                    if len(potentialVisitors) == 0:
#                        print 'Len visitors equal zero'
#                    visitedAges = []
#                    cond = False
#                    while cond == False:
#                        ageClass = np.random.choice(range(int(self.p['interactionAgeClasses'])))
#                        if ageClass not in visitedAges:
#                            visitedAges.append(ageClass)
#                        potentialVisitors = [x for x in agents if x.interactionAgeClass == ageClass and x not in venue.occupants]
#                        if len(potentialVisitors) > 0:
#                            cond = True
#                        if len(potentialVisitors) == 0 and len(visitedAges) == int(self.p['interactionAgeClasses']):
#                            print 'Error: all the age classes have zero potential visitors!'
#                            sys.exit()
#                
#                if len(potentialVisitors) > 0:
#                    
#                    distances = [self.manhattanDistance(x.house, venue) for x in potentialVisitors]
#                    
#                    meanDist = np.mean(distances)
#                    if meanDist <= 0:
#                        meanDist = 1.0
#                    newDist = []
#                    for x in distances:
#                        if x == 0:
#                            newDist.append(meanDist)
#                        else:
#                            newDist.append(x)
#                    distances = [x for x in newDist]
#                    
#                    weights = [1.0/np.power(x, self.p['distanceBetaVenue']) for x in distances]
#                    
#                    probs = [x/sum(weights) for x in weights]
#                    
#                    agent = np.random.choice(potentialVisitors, p = probs)
#                    lastRemCont = remainingContacts
#                    remainingContacts -= 2*int(math.ceil(math.pow(venue.numVirtualOccupants, self.p['venueBetaExp'])))
#                    
#                    if remainingContacts == lastRemCont:
#                        print 'Error: rem contacts do not diminish!'
#                        print remainingContacts
#                        print venue.numVirtualOccupants
#                        print len(potentialVisitors)
#                        sys.exit()
#                    
#                    venue.numVirtualOccupants += 1
#                    if agent.hospitalized == False:
#                        if self.p['behaviourModuleOn'] == True:
#                            probAttendance = self.computeAttendanceProb(agent, venue)
#                            if np.random.random() < probAttendance:
#                                venue.occupants.append(agent)
#                            else:
#                                self.cumulatedAbsences += 1
#                        else:
#                            venue.occupants.append(agent)
#                        
#                    numContacts = int(math.ceil(math.pow(len(venue.occupants)-1, self.p['venueBetaExp'])))
#                    otherAttendants = [x for x in venue.occupants if x != agent]
#                    newContacts = np.random.choice(otherAttendants, numContacts, replace = False)
#                    i = agent.interactionAgeClass
#                    j = agent.incomeQuintile
#                    agentClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
#                    effectiveContacts[i][j] += numContacts
#                    for attendant in newContacts:
#                        agentClass.actualContactsByAge[attendant.interactionAgeClass] += 1
#                        i = attendant.interactionAgeClass
#                        j = attendant.incomeQuintile
#                        attendantClass = [x for x in self.classesContacts if x.age == i and x.quintile == j][0]
#                        attendantClass.actualContactsByAge[agent.interactionAgeClass] += 1
#                        effectiveContacts[i][j] += 1
#    
#        
#        for venue in self.map.allVenues:
##            actualVisitors = []
##            for agent in venue.occupants:
##                probAttendance = self.computeAttendanceProb(agent, len(venue.occupants))
##                if np.random.random() < agent.contactReductionRate:
##                    actualVisitors.append(agent)
##            venue.occupants = [x for x in actualVisitors]
#            venue.pastAttendants.append(len(venue.occupants))
#            # Update past attendance
#            
#            attendancesDiscounted = 0
#            den = 0
#            for i in range(len(venue.pastAttendants)):
#                index = len(venue.pastAttendants)-1
#                attendancesDiscounted += venue.pastAttendants[index-i]*np.power(self.p['timeDiscountingFactor'], i)
#                den += np.power(self.p['timeDiscountingFactor'], i)
#            venue.pastAttendance = float(attendancesDiscounted)/float(den)
    
    def updateVenuesDistribution(self):
        
        
        self.totalAttendants = 0
        for venue in self.map.allVenues:
            
            # print 'Venue attendants: ' + str(len(venue.occupants))
            self.totalAttendants += len(venue.occupants)
            # Update size category
            if len(venue.occupants) <= 20:
                venue.sizeCat = len(venue.occupants)
            elif len(venue.occupants) > 20 and len(venue.occupants) <= 25:
                venue.sizeCat = 21
            elif len(venue.occupants) > 25 and len(venue.occupants) <= 30:
                venue.sizeCat = 22
            elif len(venue.occupants) > 30 and len(venue.occupants) <= 35:
                venue.sizeCat = 23
            elif len(venue.occupants) > 35 and len(venue.occupants) <= 40:
                venue.sizeCat = 24
            elif len(venue.occupants) > 40 and len(venue.occupants) <= 50:
                venue.sizeCat = 25
            else:
                venue.sizeCat = 26
                
            self.venuesAttendantsDist[venue.sizeCat-1] += 1
        
        # Update mean attendance by town
        for town in self.map.towns:
            townAttendances = [len(x.occupants) for x in town.venues]
            town.meanAttendance = 0
            if sum(townAttendances) > 0:
                town.meanAttendance = np.mean(townAttendances)
    
        print 'Total attendances: ' + str(self.totalAttendants)
        if self.totalAttendants < self.minAttendances:
            self.minAttendances = self.totalAttendants
        print 'Min attendaces: ' + str(self.minAttendances)
        self.venuesAttendances.append(self.totalAttendants)
        if sum(self.venuesAttendantsDist) > 0:
            self.shareAttendances = [float(x)/sum(self.venuesAttendantsDist) for x in self.venuesAttendantsDist]
            
    def updateIsolationRates(self, day):
        
        allExpectedCosts = []
        isolationRates = []
        infectionCosts = []
        ipFactors = []
        maxIsolationRates = []
        zeroIsolationRatesShare = 0
        totAgents = 0
        zeroIsolationAgents = 0
        for house in self.map.occupiedHouses:
            # houseIsolationRates = []
            adults = [x for x in house.occupants if x.age >= 16]
            for agent in house.occupants:
                totAgents += 1
                # The isolation rate is computed for each household's member
                # Depending on age and experiences, each member has a different isolation rate
                if agent.testPositive == True:
                    agent.isolationRate = 1.0
                    agent.maxIsolationRate = 1.0
                else:
                    # - Compute expected total cost
                    if agent.probInfection > 0:
                        ipFactor = (1.0-agent.probInfection)/agent.probInfection
                        
                        ipFactors.append(ipFactor)
                        
                        den = np.power(1.0+self.p['infectionProbFactor']*ipFactor, self.p['infectionProbExp'])
                        expectedTotalCost = agent.costInfection/den
                        
                        allExpectedCosts.append(expectedTotalCost)
            
                        incomeFactor = float(agent.incomeQuintile+1.0)/5.0
                        
                        agent.isolationRate = (1.0-np.exp(-1*self.p['betaIsolationRate']*incomeFactor*expectedTotalCost))
                        agent.maxIsolationRate = (1.0-np.exp(-1*self.p['betaIsolationRate']*expectedTotalCost))
                    else:
                        zeroIsolationAgents += 1
                        agent.isolationRate = 0.0
                        agent.maxIsolationRate = 0.0

                # houseIsolationRates.append(agent.isolationRate)
                isolationRates.append(agent.isolationRate)
                infectionCosts.append(agent.costInfection)
                
                maxIsolationRates.append(agent.maxIsolationRate)
            # The household isolation rate is the highest isolation rate
#            house.isolationRate = np.mean(houseIsolationRates)
#            for member in [x for x in adults if x.testPositive == False]:
#                member.isolationRate += (house.isolationRate-member.isolationRate)*self.p['houseConformRate']
#                member.isolationRates.append(member.isolationRate)
        
        # Debugging code
        zeroIsolationRatesShare = 0
        if totAgents > 0:
            zeroIsolationRatesShare = float(zeroIsolationAgents)/float(totAgents)
        print 'Share of agents who do not isolate: ' + str(zeroIsolationRatesShare)
        
        print 'Mean isolation rate: ' + str(np.mean(isolationRates))
        print 'Median isolation rate: ' + str(np.median(isolationRates))
        
        print 'Mean max isolation rate: ' + str(np.mean(isolationRates))
        print 'Median max isolation rate: ' + str(np.median(isolationRates))
        
        print 'Mean cost: ' + str(np.mean(maxIsolationRates))
        print 'Median cost: ' + str(np.median(maxIsolationRates))
        
        if len(ipFactors) > 0:
            print 'Mean ip: ' + str(np.mean(ipFactors))
            print 'Median ip: ' + str(np.median(ipFactors))
            
        
        
        if len(allExpectedCosts) > 0:
            print 'Max expected cost (isolation rate): ' + str(max(allExpectedCosts))
            print 'Median expected cost (isolation rate): ' + str(np.median(allExpectedCosts))
            # pdb.set_trace()
      
        
    def socialNorm(self):
        adults = [x for x in self.pop.livingPeople if x.age >= 16 and x.testPositive == False and x.hospitalized == False]
        weights = []
        for agent in adults:
            agent.tempIR = agent.isolationRate
            friends = [x for x in agent.socialContacts.neighbors(agent) if x.age >= 16 and x.testPositive == False and x.hospitalized == False]
            if len(friends) > 0:
                irs = [x.isolationRate for x in friends]
                weights = [agent.socialContacts[agent][x]['weight'] for x in friends]
                expWeights = [np.exp(self.p['socialWeightExp']*x) for x in weights]
                den = sum(expWeights)
                num = sum([x*y for x, y in zip(irs, expWeights)])
                weightedIR = float(num)/float(den)
                deltaRate = weightedIR-agent.tempIR
                agent.tempIR += deltaRate*(1.0-np.exp(-1*self.p['socialNormSensitivity']))
            
        print 'Max weights: ' + str(max(weights))
        print 'Mean weights: ' + str(np.mean(weights))
        print 'Median weights: ' + str(np.median(weights))
         
        for agent in adults:
            agent.isolationRate = agent.tempIR
        
        vulnerabilityIndexes = []
        for house in self.map.occupiedHouses:
            adults = [x for x in house.occupants if x.age >= 16]
            maxIR = np.mean([max(x.maxIsolationRate, x.isolationRate) for x in adults])
            house.isolationRate = np.mean([x.isolationRate for x in adults])
            
            vulnerabilityIndexes.append(house.vulnerabilityIndex)
            
            # Isolation rate variation due to the provision of services between relatives
            if house.vulnerabilityIndex > 0.0:
                # In this case, increase the isolation rate by a factor proportional to the vf
                house.isolationRate += (maxIR-house.isolationRate)*(1.0-np.exp(-1*self.p['vulnerabilityIsolationExp']*house.vulnerabilityIndex))
            else:
                # In this case, the isolation rate is decreased
                house.isolationRate -= house.isolationRate*(1.0-np.exp(self.p['vulnerabilityIsolationExp']*house.vulnerabilityIndex))
                
            for member in [x for x in adults if x.testPositive == False]:
                member.isolationRate += (house.isolationRate-member.isolationRate)*self.p['houseConformRate']
                member.isolationRates.append(member.isolationRate)
            children = [x for x in house.occupants if x not in adults]
            for child in children:
                child.isolationRate = house.isolationRate
        
        print 'Max vulnerability indexes: ' + str(max(vulnerabilityIndexes))
        print 'Median vulnerability indexes: ' + str(np.median(vulnerabilityIndexes))
        print 'Mean vulnerability indexes: ' + str(np.mean(vulnerabilityIndexes))
        print 'Min vulnerability indexes: ' + str(min(vulnerabilityIndexes))
        
        
        for i in range(5):
            
            reductionRatesInQuintile = [x.isolationRate for x in self.pop.livingPeople if x.incomeQuintile == i]
            print 'Quintile ' + str(i) + ' mean reduction rate:' + str(np.mean(reductionRatesInQuintile))
            print 'Quintile ' + str(i) + ' max reduction rate:' + str(max(reductionRatesInQuintile))
            print 'Quintile ' + str(i) + ' min reduction rate:' + str(min(reductionRatesInQuintile))
            print ''
            if i == 0:
                self.mobilityReduction_Q1 = np.mean(reductionRatesInQuintile)
            if i == 1:
                self.mobilityReduction_Q2 = np.mean(reductionRatesInQuintile)
            if i == 2:
                self.mobilityReduction_Q3 = np.mean(reductionRatesInQuintile)
            if i == 3:
                self.mobilityReduction_Q4 = np.mean(reductionRatesInQuintile)
            if i == 4:
                self.mobilityReduction_Q5 = np.mean(reductionRatesInQuintile)
            
        for i in range(int(self.p['ageClasses'])):
            reductionRatesByAge = [x.isolationRate for x in self.pop.livingPeople if x.ageClass == i]
            if i == 0:
                self.mobilityReduction_A1 = np.mean(reductionRatesByAge)
            if i == 1:
                self.mobilityReduction_A2 = np.mean(reductionRatesByAge)
            if i == 2:
                self.mobilityReduction_A3 = np.mean(reductionRatesByAge)
            if i == 3:
                self.mobilityReduction_A4 = np.mean(reductionRatesByAge)
            if i == 4:
                self.mobilityReduction_A5 = np.mean(reductionRatesByAge)
            if i == 5:
                self.mobilityReduction_A6 = np.mean(reductionRatesByAge)
            if i == 6:
                self.mobilityReduction_A7 = np.mean(reductionRatesByAge)
            if i == 7:
                self.mobilityReduction_A8 = np.mean(reductionRatesByAge)
            if i == 8:
                self.mobilityReduction_A9 = np.mean(reductionRatesByAge)
        
            
    def socialLearning_Cond(self, person, probCondition, event):
        
        prob = probCondition
        house = person.house
        events = [x for x in house.periodEvents if x.condition == event]
        houseEvents = [x for x in house.inHouseEvents if x.condition == event]
        friendsWithEvents = [x.agent for x in events]
                    
        membersWithEvents = [x.agent for x in houseEvents if x != person]
        # membersWithNoEvents = [x for x in house.occupants if x.hadPandemicEvent == False and x != person]
        
        allNeighbors = friendsWithEvents+membersWithEvents #+membersWithNoEvents+friendsWithNoEvents
        if len(allNeighbors) > 0:
            np.random.shuffle(allNeighbors)
            for agent in allNeighbors:
                # Implement change according to observation of friend's events
                # if agent.hadPandemicEvent == True:
                distance = 0
                if agent in friendsWithEvents:
                    singleEvent = [x for x in events if x.agent == agent][0]
                    distance = singleEvent.distanceWeight
                weight = np.exp(self.p['riskDistSensitivity']*distance)
                deltaAge = person.age-agent.age
                # The event will have the greater effect the more the infected person was younger
                expRate = np.exp(self.p['deltaAgeExp']*deltaAge)/(np.exp(self.p['deltaAgeExp']*deltaAge)+1.0)
                prob += (1.0-prob)*expRate/weight
        else: # In this case, the probability is reduced
            prob -= prob*(1.0-np.exp(-1*self.p['fearExtintionRate']))
                
        return prob
        
    def computeAttendanceProb(self, agent, venue):
        
        actualAttendants = len(venue.occupants)
        numAttendandants = int(math.ceil(self.p['actualAttendantsWeight']*float(actualAttendants)+(1.0-self.p['actualAttendantsWeight'])*venue.pastAttendance))
        
        if numAttendandants <= 20:
            sizeCat = numAttendandants
        elif numAttendandants > 20 and numAttendandants <= 25:
            sizeCat = 21
        elif numAttendandants > 25 and numAttendandants <= 30:
            sizeCat = 22
        elif numAttendandants > 30 and numAttendandants <= 35:
            sizeCat = 23
        elif numAttendandants > 35 and numAttendandants <= 40:
            sizeCat = 24
        elif numAttendandants > 40 and numAttendandants <= 50:
            sizeCat = 25
        else:
            sizeCat = 26
        
        behaviouralIsolation = (1.0 - agent.house.contactReductionRates[sizeCat-1])
        prob = min(agent.contactReductionRate, behaviouralIsolation)
        
        return prob
    
    
#    def socialInteraction(self):
#        # In this function, the agent draws from his networks a number of daily contacts which depends on his isolation behaviour.
#        # The probability that a node is selected as daily contact depends on the links' weights,
#        
#        ## print 'Class Contacts Matrix: ' + str(self.classContactsMatrix)
#        
#        susceptible = [x for x in self.pop.livingPeople if x.healthStatus == 'susceptible']
#        for agent in susceptible:
#            agent.dailyContacts = []
#            agent.randomContacts = []
#            # 1 - Get daily contacts
#            agent.numContacts = 0
#            friends = [x for x in agent.socialContacts.nodes() if x != agent and x.dead == False and x.hospitalized == False and (agent.socialContacts[agent][x]['weight']*x.contactReductionRate) > 0]
#            if len(friends) > 0:
#                quintile = agent.incomeQuintile
#                if self.p['5yearAgeClasses'] == False:
#                    ageClasses = int(self.p['ageClasses'])
#                    agentAgeGroup = agent.ageClass
#                else:
#                    ageClasses = int(self.p['interactionAgeClasses'])
#                    agentAgeGroup = agent.interactionAgeClass
#                averageContacts = sum([self.classContactsMatrix[agentAgeGroup][x][quintile] for x in range(ageClasses)])*self.p['dailyContactsShare']
#                if self.lockdown == True:
#                    averageContacts *= self.p['lockdownContactReductionRate']
#                
##                if agent.age >= 65:
##                    averageContacts *= 3.0
#                
##                den = sum([agent.socialContacts[agent][x]['weight'] for x in friends])
##                num = sum([agent.socialContacts[agent][x]['weight']*x.contactReductionRate for x in friends])
##                averageIsolationRate = 0
##                if den > 0:
##                    averageIsolationRate = num/den
##                meanNumContacts = float(averageContacts)*agent.contactReductionRate*averageIsolationRate
#                    
#                agent.numContacts = np.random.poisson(averageContacts)
#                
#                agent.numRandomContacts = int(float(agent.numContacts)*self.p['randomContactsShare'])
#                # Now, the contacts are randomly sampled from the agent's social network
#                if agent.numContacts > len(friends):
#                    agent.numContacts = len(friends)
#    
#        # totContacts = []
#        
#        # Once the agents' number of daily contacts has been computed, the actual contacts are randomlyb drawn from the network of friends.
#        random.shuffle(susceptible)
#        
#        for agent in susceptible:
#            friends = [x for x in agent.socialContacts.nodes() if x != agent and x not in agent.dailyContacts and x.dead == False and x.hospitalized == False and (agent.socialContacts[agent][x]['weight']*x.contactReductionRate) > 0 and x.numContacts > len(x.dailyContacts)]
#            if agent.numContacts > len(agent.dailyContacts) and len(friends) > 0:
#                additionalContacts = agent.numContacts-len(agent.dailyContacts)
#                if additionalContacts > len(friends):
#                    additionalContacts = len(friends)
#                isolationRates = [x.contactReductionRate for x in friends]
#                townIndexes = []
#                for friend in friends:
#                    if friend.house.town == agent.house.town:
#                        townIndexes.append(self.p['sameTownWeight'])
#                    else:
#                        townIndexes.append(1.0)
#                linksWeights = [agent.socialContacts[agent][x]['weight'] for x in friends]
#                weights = [w*r*t for w, r, t in zip(linksWeights, isolationRates, townIndexes)]
#                contactsProbs = [x/sum(weights) for x in weights]
#                metFriends = np.random.choice(friends, additionalContacts, replace = False, p = contactsProbs)
#                agent.dailyContacts.extend(metFriends)
#                
#                for friend in metFriends:
#                    friend.dailyContacts.append(agent)
#        
#        outOfTownContacts = 0
#        outOfTownFriends = 0
#        totFriends = 0
#        totContacts = 0
#        for agent in susceptible:
#            
#            # print 'Num daily contacts: ' + str(len([x for x in agent.dailyContacts]))
#            
#            totFriends += agent.socialContacts.number_of_nodes()
#            totContacts += len([x for x in agent.dailyContacts])
#            outOfTownFriends += len([x for x in agent.socialContacts.nodes() if x.house.town != agent.house.town])
#            outOfTownContacts += len([x for x in agent.dailyContacts if x.house.town != agent.house.town])
#        
#        print 'Over ' + str(len(susceptible)) + ' susceptible, there are ' + str(outOfTownFriends) + ' out-of-town friends'
#        print 'Over ' + str(len(susceptible)) + ' susceptible, there are ' + str(outOfTownContacts) + ' out-of-town contacts'
#        
#        print 'Share of out-of-town friends: ' + str(float(outOfTownFriends)/float(totFriends))
#        print 'Share of out-of-town contacts: ' + str(float(outOfTownContacts)/float(totContacts))
                    
        ### Add random contacts
#        for agent in susceptible:
#            agent.currentTown = agent.house.town
#            towns = [x.house.town for x in agent.dailyContacts]
#            commutingTowns = [x for x in towns if x != agent.house.town]
#            if len(commutingTowns) > 0:
#                if len(list(set(commutingTowns))) > 1:
#                    commutingTown = np.random.choice(commutingTowns)
#                else:
#                    commutingTown = commutingTowns[0]
#                agent.currentTown = commutingTown
#                excludedTowns = [x for x in commutingTowns if x != agent.currentTown]
#                contacts = [x for x in agent.dailyContacts if x.house.town not in excludedTowns]
#                agent.dailyContacts = [x for x in contacts]
#            else:
#                agent.currentTown = agent.house.town
#            
#            residualPop = [x for x in self.pop.livingPeople if x.contactReductionRate > 0 and x.numRandomContacts > len(x.randomContacts) and (x.house.town == agent.currentTown or x.house.town == agent.house.town)]
#            # Random contacts depend only in distance
#            if agent.numRandomContacts > len(agent.randomContacts) and len(residualPop) > 0:
#                additionalContacts = (agent.numRandomContacts)-len(agent.randomContacts)
#                if len(residualPop) < additionalContacts:
#                    additionalContacts = len(residualPop)
#                isolationRates = [x.contactReductionRate for x in residualPop]
#                geoDistances = [self.geoDistance(agent.house, x.house) for x in residualPop]
#                distanceExps = [self.p['locationInteractionBeta']*math.pow(x, self.p['locationInteractionExp']) for x in geoDistances]
#                distancesWeights = [1.0/math.exp(x) for x in distanceExps]
#                weights = [w*r for w, r in zip(distancesWeights, isolationRates)]
#                contactsProbs = [x/sum(weights) for x in weights]
#                randContacts = np.random.choice(residualPop, additionalContacts, replace = False, p = contactsProbs)
#                agent.randomContacts.extend(randContacts)
#
#                for contact in randContacts:
#                    contact.randomContacts.append(agent)
#                    
#            totContacts.append(agent.numContacts)
#        
#        print 'Average num of contacts: ' + str(np.mean(totContacts))
                    
    def townDistance(self, agentTown, contactTown):
        agentX = agentTown.x
        agentY = agentTown.y
        contactX = contactTown.x
        contactY = contactTown.y
        xDist = abs(agentX - contactX)
        yDist = abs(agentY - contactY)
        return xDist + yDist
        
    def geoDistance(self, agentHouse, contactHouse):
        d = self.p['townGridDimension']
        agentX = agentHouse.town.x*d+agentHouse.x
        agentY = agentHouse.town.y*d+agentHouse.y 
        contactX = contactHouse.town.x*d+contactHouse.x
        contactY = contactHouse.town.y*d+contactHouse.y 
        xDist = abs(agentX - contactX)
        yDist = abs(agentY - contactY)
        return xDist + yDist
    
    def exposureProcess(self, day):
        
        print 'Probs deaths if intubated: ' + str(self.probsDeathIntubated)
        self.newExposed = 0
        self.infectedByClass = [0]*int(self.p['incomeClasses'])
        self.infectedByAge = [0]*int(self.p['ageClasses'])
        # self.newCases = 0
        ### Exogenous exposure: some agents are exposed from outside  ######
        exposedAgents = []
        internationalExposed = []
        if self.lockdown == False:
            travellers = [x for x in self.pop.livingPeople if x.healthStatus == 'susceptible' and x.age >= 18]
            exogenouslyInfected = int(round(float(len(travellers))*self.p['exogenousInfectionRate']))
            if exogenouslyInfected > 0:
                totalClassWeight = 0
                for i in range(int(self.p['numberClasses'])):
                    totalClassWeight += np.exp(self.p['classWeightParam']*i)
                classFactors = [np.exp(self.p['classWeightParam']*x.classRank)/totalClassWeight for x in travellers]
                totalHouses = float(len(self.map.allHouses))
                townFactor = [float(len(x.house.town.houses))/totalHouses for x in travellers]
                weights = [a*b for a, b in zip(classFactors, townFactor)]
                probs = [x/sum(weights) for x in weights]
                internationalExposed = list(np.random.choice(travellers, size=exogenouslyInfected, replace=False, p=probs))
                
                for agent in internationalExposed:
                    newNode = Node(agent.id, agent.age, agent.incomeQuintile)
                    self.infectionsNetwork.add_node(newNode)
                
                print 'Exogenous infections: ' + str(exogenouslyInfected)
            
                exposedAgents.extend(internationalExposed)
                    
        ##### Endogenous exposure: exposure from contagious agents in the population #######
        
        ### For every susceptible agent, the probability to be infected through communal or domestic contacts is computed
        internalSusceptible = [x for x in self.pop.livingPeople if x not in internationalExposed]
        susceptible = [x for x in internalSusceptible if x.healthStatus == 'susceptible']
      
        num1to18 = 0
        numberContacts1to18 = 0
        averageContagiousness1to18 = 0
        careRiskFactor1to18 = 0
        homeRiskFactor1to18 = 0
        
        numOver65 = 0
        numberContactsOver65 = 0
        averageContagiousnessOver65 = 0
        careRiskFactorOver65 = 0
        homeRiskFactorOver65 = 0
        
        for person in susceptible:
            
            # Social infection
            venues = [x for x in self.map.allVenues if person in x.occupants]
            infected = False
            for venue in venues:
                if infected == False:
                    contacts = [x for x in venue.occupants if x != person]
                    if len(contacts) > 0:
                        numContacts = int(math.ceil(math.pow(len(contacts), self.p['venueBetaExp'])))
                        actualContacts = np.random.choice(contacts, numContacts, replace=False)
                        beta = self.p['betaCommunity'] #/math.pow(len(contacts), self.p['venueBetaExp'])
                        for contact in actualContacts:
                            if contact.healthStatus == 'infectious':
                                prob = beta*contact.contagiousnessIndex
                                if np.random.random() < prob:
                                    infected = True
                                    exposedAgents.append(person)
                                    newNode = Node(person.id, person.age, person.incomeQuintile)
                                    vectorNode = [x for x in self.infectionsNetwork.nodes() if x.agentID == contact.id][0]
                                    self.infectionsNetwork.add_edge(vectorNode, newNode, color = "r")
                                    break
                else:
                    break
            
            # Domestic infection
            if person not in exposedAgents:
                householdInfectionIndex = self.p['betaHousehold']*person.domesticRiskFactor
                probDomesticInfection = (math.exp(householdInfectionIndex)-1.0)/math.exp(householdInfectionIndex)
                if np.random.random() < probDomesticInfection:
                    exposedAgents.append(person)
                    infectedHouseholds = [x for x in person.house.occupants if x.healthStatus == 'infectious' and x.hospitalized == False]
                    vector = np.random.choice(infectedHouseholds)
                    newNode = Node(person.id, person.age, person.incomeQuintile)
                    vectorNode = [x for x in self.infectionsNetwork.nodes() if x.agentID == vector.id][0]
                    self.infectionsNetwork.add_edge(vectorNode, newNode, color = "g")
        
        for person in exposedAgents:
        
            self.newExposed += 1
            self.infectedByClass[person.incomeQuintile] += 1
            self.infectedByAge[person.ageClass] += 1
            person.healthStatus = 'exposed'
            person.hasBeenExposed = True
            person.daysFromInfection = 0
            # An incupbation period is assigned (before the symptoms appear)
            # Agents become infectious 2 days before the appearance of symptoms.
            person.incubationPeriod = max(int(math.ceil(np.random.lognormal(self.p['meanIncubation'], self.p['sdIncubation']))), self.p['minIncubation'])
            person.viralLoad = np.random.random()
            person.contagiousnessIndex = math.pow(person.viralLoad, self.p['loadContagiousnessExp'])
            genderIndex = 0
            if person.sex == 'female':
                genderIndex = 1

            prob = self.probSymptomatic[person.ageClass][person.incomeQuintile][genderIndex]
            
#            print ''
#            print 'Person id: ' + str(person.id)
#            print 'Person age class: ' + str(person.ageClass)
#            print 'Person income quintile: ' + str(person.incomeQuintile)
#            print 'Person gender index: ' + str(genderIndex)
#            print 'Prob symptomatic: ' + str(prob)
            
            # With a certain probability the agent is asymtpomatic
            if np.random.random() > prob: # self.p['probSymptomatic'][person.ageClass]:
                
                # print 'The person ' + str(person.id) + ' is asymptomatic'
                # In this case the agent is asymptomatic
                person.severityLevel = 1
                person.symptomsLevel = self.p['symptomsLevels'][person.severityLevel-1]
                person.recoveryPeriod = np.random.choice(self.recoveryPeriods[person.severityLevel-1])
            else:
                # In this case the agent is not asymptomatic.
                # It may develop mild symptoms, severe symtoms (hospitalized) pr critical symptoms (intensive care)
                # print 'The person ' + str(person.id) + ' is not asymptomatic'
                
                prob = self.probsHospitalization[person.ageClass][person.incomeQuintile][genderIndex]
                
                # print 'Prob hospitalized: ' + str(prob)
                
                if np.random.random() > prob: # self.p['probsHospitalization'][person.ageClass]:
                    # In this case, agent is NOT hospitalized (i..e develops mild conditions)
                    
                    # print 'The person ' + str(person.id) + ' is not hospitalized'
                    
                    person.severityLevel = 2
                    person.symptomsLevel = self.p['symptomsLevels'][person.severityLevel-1]
                    person.recoveryPeriod = np.random.choice(self.recoveryPeriods[person.severityLevel-1])
                else:   ## In this case is hospitalized
                    
                    # print 'The person ' + str(person.id) + ' is hospitalized'
                    
                    prob = self.probsIntensiveCare[person.ageClass][person.incomeQuintile][genderIndex]
                    
                    # print 'Prob icu: ' + str(prob)
                    
                    if np.random.random() > prob: # self.p['probsIntensiveCare'][person.ageClass]:
                        # In this case, agent is NOT in intensive care (i.e. just hospitalized)
                        
                        # print 'The person ' + str(person.id) + ' is not intubated'
                        
                        person.severityLevel = 3
                        person.symptomsLevel = self.p['symptomsLevels'][person.severityLevel-1]
                        person.recoveryPeriod = np.random.choice(self.recoveryPeriods[person.severityLevel-1])
                        
                    else:
                        
                        # print 'The person ' + str(person.id) + ' is intubated'
                        
                        # In this case, agent is in icu
                        person.severityLevel = 4
                        person.symptomsLevel = self.p['symptomsLevels'][person.severityLevel-1]
                        person.recoveryPeriod = np.random.choice(self.recoveryPeriods[person.severityLevel-1])
        
        if num1to18 > 0:
            print 'Average contacts 1/18-year-old: ' + str(float(numberContacts1to18)/float(num1to18))
            print 'Average contagiousness 1/18-year-old: ' + str(float(averageContagiousness1to18)/float(num1to18))
            print 'Average care risk 1/18-year-old: ' + str(float(careRiskFactor1to18)/float(num1to18))
            print 'Average home risk 1/18-year-old: ' + str(float(homeRiskFactor1to18)/float(num1to18))
        
        if numOver65 > 0:
            print 'Average contacts over65-year-old: ' + str(float(numberContactsOver65)/float(numOver65))
            print 'Average contagiousness over65-year-old: ' + str(float(averageContagiousnessOver65)/float(numOver65))
            print 'Average care risk over65-year-old: ' + str(float(careRiskFactorOver65)/float(numOver65))
            print 'Average home risk over65-year-old: ' + str(float(homeRiskFactorOver65)/float(numOver65))
        
        # Some agents die with a certain infection-fatality ratio.
        
        newMild = [x for x in exposedAgents if x.symptomsLevel == 'mild']
        numMild = len(newMild)
        # Assign severity of symptoms to agents with mild condition
        mildConditionIndex = [x for x in np.random.exponential(self.p['mildExponentialPar'], size=numMild)]
        mildConditionIndex.sort(reverse=True)
        mildConditionIndex = [x/max(mildConditionIndex) for x in mildConditionIndex]
        agentsToAssign = list(newMild)
        n = 0
        while len(agentsToAssign) > 0:
            loadFactors = [np.exp(x.viralLoad*self.p['viralLoadWeight']) for x in agentsToAssign]
            ageFactors = [self.p['infectionFatalityRatio'][x.ageClass] for x in agentsToAssign]
            # classFactors = [1.0/np.exp((x.incomeQuintile)*self.p['incomeSeverityWeight']) for x in agentsToAssign]
            classFactors = [math.pow(self.p['severityClassBias'], x.incomeQuintile) for x in agentsToAssign]
            genderFactors = []
            for agent in agentsToAssign:
                if agent.sex == 'male':
                    genderFactors.append(1.0)
                else:
                    genderFactors.append(self.p['severityGenderBias'])
            # genderFactors = [x.genderWeight for x in agentsToAssign]
            weights = [a*b*c*d for a, b, c, d in zip(loadFactors, ageFactors, classFactors, genderFactors)]
            probs = [x/sum(weights) for x in weights]
            mildAgent = np.random.choice(agentsToAssign, p=probs)
            mildAgent.mildConditionIndex = mildConditionIndex[n]
            mildAgent.contactReductionRate = math.pow(1.0 - mildAgent.mildConditionIndex, self.p['mildMobilityExp'])
            n += 1
            agentsToAssign.remove(mildAgent)
        
        dead = []
        for y in range(int(self.p['ageClasses'])):
            for i in range(int(self.p['incomeClasses'])):
                # Males
                males = [x for x in exposedAgents if x.ageClass == y and x.incomeQuintile == i and x.sex == 'male']
                popMales = len(males)
                # Assign deaths to conditions
                # deaths = int(math.ceil(float(popMales)*self.infectionFatalityRatio[y][i][0]))
                deaths = np.random.poisson(float(popMales)*self.infectionFatalityRatio[y][i][0])
                residualDeaths = deaths
                # Intubated
                intubated = [x for x in males if x.severityLevel == 4]
                deathIntubated = int(math.ceil(float(residualDeaths)*self.p['shareDeathByCondition']))
                if deathIntubated < len(intubated):
                    intubatedDead = np.random.choice(intubated, deathIntubated, replace = False)
                    dead.extend(intubatedDead)
                    intubatedDeaths = deathIntubated
                else:
                    dead.extend([x for x in intubated])
                    intubatedDeaths = len(intubated)
                residualDeaths -= intubatedDeaths
                
                # Hospitalized
                hospitalized = [x for x in males if x.severityLevel == 3]
                deathHospitalized = int(math.ceil(float(residualDeaths)*self.p['shareDeathByCondition']))
                if deathHospitalized < len(hospitalized):
                    hospitalizedDead = np.random.choice(hospitalized, deathHospitalized, replace = False)
                    dead.extend(hospitalizedDead)
                    hospitalizedDeaths = deathHospitalized
                else:
                    dead.extend([x for x in hospitalized])
                    hospitalizedDeaths = len(hospitalized)
                residualDeaths -= hospitalizedDeaths
                
                # Sample deaths among symptomatic not hospitalized
                # Proportional to the mildConditionIndex
                if residualDeaths > 0:
                    symptomatic = [x for x in males if x.severityLevel == 2]
                    if residualDeaths < len(symptomatic):
                        symptomatic.sort(key=operator.attrgetter("mildConditionIndex"), reverse=True)
                        dead.extend(symptomatic[:residualDeaths])
                    else:
                        dead.extend([x for x in symptomatic])
                
                # Females
                females = [x for x in exposedAgents if x.ageClass == y and x.incomeQuintile == i and x.sex == 'female']
                popFemales = len(females)
                # Assign deaths to conditions
                # deaths = int(math.ceil(float(popFemales)*self.infectionFatalityRatio[y][i][1]))
                deaths = np.random.poisson(float(popFemales)*self.infectionFatalityRatio[y][i][1])
                residualDeaths = deaths
                # Intubated
                intubated = [x for x in females if x.severityLevel == 4]
                deathIntubated = int(math.ceil(float(residualDeaths)*self.p['shareDeathByCondition']))
                if deathIntubated < len(intubated):
                    intubatedDead = np.random.choice(intubated, deathIntubated, replace = False)
                    dead.extend(intubatedDead)
                    intubatedDeaths = deathIntubated
                else:
                    dead.extend([x for x in intubated])
                    intubatedDeaths = len(intubated)
                residualDeaths -= intubatedDeaths
                
                # Hospitalized
                hospitalized = [x for x in females if x.severityLevel == 3]
                deathHospitalized = int(math.ceil(float(residualDeaths)*self.p['shareDeathByCondition']))
                if deathHospitalized < len(hospitalized):
                    hospitalizedDead = np.random.choice(hospitalized, deathHospitalized, replace = False)
                    dead.extend(hospitalizedDead)
                    hospitalizedDeaths = deathHospitalized
                else:
                    dead.extend([x for x in hospitalized])
                    hospitalizedDeaths = len(hospitalized)
                residualDeaths -= hospitalizedDeaths
                
                # Sample deaths among symptomatic not hospitalized
                # Proportional to the mildConditionIndex
                if residualDeaths > 0:
                    symptomatic = [x for x in females if x.severityLevel == 2]
                    if residualDeaths < len(symptomatic):
                        symptomatic.sort(key=operator.attrgetter("mildConditionIndex"), reverse=True)
                        dead.extend(symptomatic[:residualDeaths])
                    else:
                        dead.extend([x for x in symptomatic])
                
        for person in dead:
            if person.severityLevel == 2:
                person.placeOfDeath = 'Home'
            elif person.severityLevel == 3:
                person.placeOfDeath = 'Hospital'
            elif person.severityLevel == 4:
                person.placeOfDeath = 'ICU'
            person.severityLevel = 5
            person.symptomsLevel = self.p['symptomsLevels'][person.severityLevel-1]
            person.recoveryPeriod = np.random.choice(self.recoveryPeriods[person.severityLevel-1])
        
        
        
        
        # totInfected = sum(self.infectedByAge)
#        totDeaths = sum(self.deathsByAge)
#        totIntubated = sum(self.intubatedByAge)
#        totHospitalized = sum(self.hospitalizedByAge)
#        totMild = sum(self.symptomaticByAge)
#        totAsymptomatic = sum(self.infectedByAge) - (totMild+totHospitalized+totIntubated+totDeaths)
#        numConditions = [totAsymptomatic, totMild, totHospitalized, totIntubated, totDeaths]
        
        for person in self.pop.livingPeople:
            # Clean daily contacts
            person.nokCareContacts = []
            person.dailyContacts = []
        
        # self.shareConditions = [float(x)/float(sum(numConditions)) for x in numConditions]
        
        print 'New exposed: ' + str(self.newExposed)
      
        for venue in self.map.allVenues:
            venue.occupants[:] = []
            venue.numVirtualOccupants = 0
        
        # venuesToDelete = np.random.choice(self.map.allVenues, self.newVenues, replace = False)
        
        
          
#        print 'Share conditions: ' + str(self.shareConditions)
#        print 'Infected by age: ' + str(self.infectedByAge)
#        print 'Infected by class: ' + str(self.infectedByClass)
#        print 'Deaths by age: ' + str(self.deathsByAge)
#        print 'Deaths by class: ' + str(self.deathsByClass)
#        print 'Intubated by age: ' + str(self.intubatedByAge)
#        print 'Intubated by class: ' + str(self.intubatedByClass)
#        print 'Hospitalized by age: ' + str(self.hospitalizedByAge)
#        print 'Hospitalized by class: ' + str(self.hospitalizedByClass)
#        print 'Mild by age: ' + str(self.symptomaticByAge)
#        print 'Mild by class: ' + str(self.symptomaticByClass)
        
#        print ''
    
    def symptomsProgression(self, day):
        
        pop = len([x for x in self.pop.livingPeople])
        pop1To11 = len([x for x in self.pop.livingPeople if x.age >= 1 and x.age < 12])
        pop12To18 = len([x for x in self.pop.livingPeople if x.age >= 12 and x.age < 19])
        pop19To39 = len([x for x in self.pop.livingPeople if x.age >= 19 and x.age < 40])
        pop40To64 = len([x for x in self.pop.livingPeople if x.age >= 40 and x.age < 65])
        popOver65 = len([x for x in self.pop.livingPeople if x.age >= 65])
        
#        print 'Share of 1/11-year-old: ' + str(100*float(pop1To11)/float(pop)) + ' (10.6)'
#        print 'Share of 12/18-year-old: ' + str(100*float(pop12To18)/float(pop)) + ' (6.8)'
#        print 'Share of 19/39-year-old: ' + str(100*float(pop19To39)/float(pop)) + ' (26.1)'
#        print 'Share of 40/64-year-old: ' + str(100*float(pop40To64)/float(pop)) + ' (35.7)'
#        print 'Share of over 65: ' + str(100*float(popOver65)/float(pop)) + ' (20.8)'
        
        # Reset variables
        self.totalHospitalized = 0
        self.totalIntubated = 0
        self.deathsByClass = [0]*int(self.p['incomeClasses'])
        self.deathsByAge = [0]*int(self.p['ageClasses'])
        self.hospitalizedByClass = [0]*int(self.p['incomeClasses'])
        self.hospitalizedByAge = [0]*int(self.p['ageClasses'])
        self.intubatedByClass = [0]*int(self.p['incomeClasses'])
        self.intubatedByAge = [0]*int(self.p['ageClasses'])
        self.symptomaticByClass = [0]*int(self.p['incomeClasses'])
        self.symptomaticByAge = [0]*int(self.p['ageClasses'])
        
        prePop = len(self.pop.livingPeople)
        susceptibles = [x for x in self.pop.livingPeople if x.healthStatus == 'susceptible'] 
        # notHospitalized = [x for x in self.pop.livingPeople if x.haveBeenHospitalized == False]
        
        #### Change state of exposed people (as determined at the time of exposure) ####
        
        infected = [x for x in self.pop.livingPeople if x not in susceptibles and x.healthStatus != 'recovered'] 
        noMoreInfectious = 0
        
#        if self.daysSinceEvent != -1:
#            self.daysSinceEvent += 1
#        if self.daysSinceEvent == self.p['daysToLockdown']:
#            self.lockdownInPlace = True
#            print 'Lockdown started in day: ' + str(day)
#        if self.daysToLockdown != -1:
#            self.daysToLockdown -= 1
        self.pastCases = self.newCases
        self.newCases = 0
        for town in self.map.towns:
            town.pastCases = town.newCases
            town.newCases = 0
        
        # For each agent hospitalized, intubated or dead, the event is added to the list of events of the neighboring households
        for house in self.map.occupiedHouses:
            house.periodEvents[:] = []
            house.inHouseEvents[:] = []
            for agent in house.occupants:
                agent.hadPandemicEvent = False
        
        for agent in infected:
            
            agent.daysFromInfection += 1
            
            if agent.healthStatus == 'exposed' and agent.daysFromInfection >= (agent.incubationPeriod-self.p['preSymptomsContagiousPeriod']):
                agent.healthStatus = 'infectious'
                agent.hasBeenInfectious = True
                self.totalInfected += 1
                
            elif agent.daysFromInfection == agent.incubationPeriod and agent.symptomsLevel != 'asymptomatic':
                # If the agents are not asymptomatic, they start to develop symptoms
                agent.symptomatic = True
                if agent.symptomsLevel == 'severe' or agent.symptomsLevel == 'critical' or agent.placeOfDeath == 'Hospital' or agent.placeOfDeath == 'ICU':
                    # Update house occupancy
                    agent.house.occupants.remove(agent)
                    agent.hadPandemicEvent = True
                    if len(agent.house.occupants) == 0:
                        if agent.house.id in self.householdIndexes:
                            self.housesNotMoreOccupied.append(agent.house.id)
                            self.resetEventsIndicators(agent.house)
                        self.map.occupiedHouses.remove(agent.house)
                        
                    self.newCases += 1
                    agent.house.town.newCases += 1
                    self.totalHospitalized += 1
                    self.cumulatedHospitalizations += 1
                    self.hospitalPopulation += 1
                    self.hospitalizedByClass[agent.incomeQuintile] += 1
                    self.totalHospitalizedByClass[agent.incomeQuintile] += 1
                    self.hospitalizedByAge[agent.ageClass] += 1
                    
                    if self.periodFirstHospitalized == -1:
                        self.periodFirstHospitalized = day
                        
                    if agent.symptomsLevel == 'critical' or agent.placeOfDeath == 'ICU':
                        self.totalIntubated += 1
                        self.cumulatedICUs += 1
                        self.icuPopulation += 1
                        self.intubatedByClass[agent.incomeQuintile] += 1
                        self.totalIntubatedByClass[agent.incomeQuintile] += 1
                        self.intubatedByAge[agent.ageClass] += 1
                    agent.hospitalized = True
                    agent.haveBeenHospitalized = True
                    agent.workingShare = 0.0
                    agent.testPositive = True
                    if agent.symptomsLevel == 'critical' or agent.placeOfDeath == 'ICU':
                        if self.periodFirstIntubated == -1:
                            self.periodFirstIntubated = day
                        agent.inIntensiveCare = True
                    
                    if agent.symptomsLevel == 'severe' or agent.placeOfDeath == 'Hospital':
                        newEvent = PandemicEvent(agent, agent.age, 'hospitalized', agent.isolationRate, 0)
                        agent.house.inHouseEvents.append(newEvent)
                        # Add event to neighbors' list of events
                        neighbors = [x for x in agent.house.infoNetwork.neighbors(agent.house)]
                        for neighbor in neighbors:
                            weight = neighbor.infoNetwork[neighbor][agent.house]['weight']
                            newEvent.distanceWeight = weight
                            neighbor.periodEvents.append(newEvent)
                        
                        if agent.age < 19:
                            self.hospitalized1To18 += 1
                            self.infected1To18 += 1
                        if agent.age >= 19 and agent.age < 40:
                            self.hospitalized19To39 += 1
                            self.infected19To39 += 1
                        if agent.age >= 40 and agent.age < 59:
                            self.hospitalized40To59 += 1
                            self.infected40To59 += 1
                        if agent.age >= 60 and agent.age < 79:
                            self.hospitalized60To79 += 1
                            self.infected60To79 += 1
                        if agent.age >= 80:
                            self.hospitalized80Over += 1
                            self.infected80Over += 1
                            
                    if agent.symptomsLevel == 'critical' or agent.placeOfDeath == 'ICU':
                        newEvent = PandemicEvent(agent, agent.age, 'intubated', agent.isolationRate, 0)
                        house.inHouseEvents.append(newEvent)
                        neighbors = [x for x in agent.house.infoNetwork.neighbors(agent.house)]
                        for neighbor in neighbors:
                            weight = neighbor.infoNetwork[neighbor][agent.house]['weight']
                            newEvent.distanceWeight = weight
                            neighbor.periodEvents.append(newEvent)
                        
                        if agent.age < 19:
                            self.icu1To18 += 1
                            self.infected1To18 += 1
                        if agent.age >= 19 and agent.age < 40:
                            self.icu19To39 += 1
                            self.infected19To39 += 1
                        if agent.age >= 40 and agent.age < 59:
                            self.icu40To59 += 1
                            self.infected40To59 += 1
                        if agent.age >= 60 and agent.age < 79:
                            self.icu60To79 += 1
                            self.infected60To79 += 1
                        if agent.age >= 80:
                            self.icu80Over += 1
                            self.infected80Over += 1
                            
#                elif agent.placeOfDeath == 'Home': ### If a person is sick at home, the public may not necessarily know that it is infected
#                    
#                    neighbors = [x for x in agent.house.infoNetwork.neighbors(agent.house)]
#                    for neighbor in neighbors:
#                        weight = neighbor.infoNetwork[neighbor][agent.house]['weight']
#                        meanIsolationRate = sum([w*r for w, r in zip(agent.house.contactReductionRates, self.shareAttendances)])
#                        newEvent = PandemicEvent(agent.age, 'infected', agent.isolationRate, meanIsolationRate, weight)
#                        neighbor.periodEvents.append(newEvent)
#                    
#                    self.newCases += 1
#                    agent.town.newCases += 1
#                    if agent.age < 19:
#                        self.infected1To18 += 1
#                    if agent.age >= 19 and agent.age < 40:
#                        self.infected19To39 += 1
#                    if agent.age >= 40 and agent.age < 59:
#                        self.infected40To59 += 1
#                    if agent.age >= 60 and agent.age < 79:
#                        self.infected60To79 += 1
#                    if agent.age >= 80:
#                        self.infected80Over += 1
                            
                else:
                    self.symptomaticByClass[agent.incomeQuintile] += 1
                    self.symptomaticByAge[agent.ageClass] += 1
                    
            elif agent.daysFromInfection == (agent.recoveryPeriod+agent.incubationPeriod):
                
                if agent.symptomsLevel == 'severe' or agent.symptomsLevel == 'critical' or agent.placeOfDeath == 'Hospital' or agent.placeOfDeath == 'ICU':
                    self.hospitalPopulation -= 1
                    if agent.symptomsLevel == 'critical' or agent.placeOfDeath == 'ICU': 
                        self.icuPopulation -= 1
                        
                if agent.symptomsLevel == 'dead':
                    agent.hadPandemicEvent = True
                    newEvent = PandemicEvent(agent, agent.age, 'death', agent.isolationRate, 0)
                    house.inHouseEvents.append(newEvent)
                    neighbors = [x for x in agent.house.infoNetwork.neighbors(agent.house)]
                    for neighbor in neighbors:
                        weight = neighbor.infoNetwork[neighbor][agent.house]['weight']
                        newEvent.distanceWeight = weight
                        neighbor.periodEvents.append(newEvent)
                    
                    
                    if agent.age < 19:
                        self.dead1To18 += 1
                    if agent.age >= 19 and agent.age < 40:
                        self.dead19To39 += 1
                    if agent.age >= 40 and agent.age < 59:
                        self.dead40To59 += 1
                    if agent.age >= 60 and agent.age < 79:
                        self.dead60To79 += 1
                    if agent.age >= 80:
                        self.dead80Over += 1
                  
                    self.deathsByClass[agent.incomeQuintile] += 1
                    self.totDeathsByClass[agent.incomeQuintile] += 1
                    self.deathsByAge[agent.ageClass] += 1
                    if self.periodFirstDeath == -1:
                        self.periodFirstDeath = day
                        
                    agent.dead = True
                    agent.hospitalized = False
                    agent.inIntensiveCare = False
                    if agent.placeOfDeath == 'Home':
                        agent.house.occupants.remove(agent)
                        if len(agent.house.occupants) == 0:
                            if agent.house.id in self.householdIndexes:
                                self.housesNotMoreOccupied.append(agent.house.id)
                                self.resetEventsIndicators(agent.house)
                            self.map.occupiedHouses.remove(agent.house)
                    if agent.partner != None:
                        agent.partner.partner = None
                    if agent.house == self.displayHouse:
                        messageString = str(self.year) + ": #" + str(agent.id) + " died aged " + str(agent.age) + "." 
                        self.textUpdateList.append(messageString)
                else:
                    if agent.symptomsLevel == 'severe' or agent.symptomsLevel == 'critical' or agent.placeOfDeath == 'Hospital' or agent.placeOfDeath == 'ICU':
                        agent.house.occupants.append(agent)
                    agent.testPositive = False
                    agent.healthStatus = 'recovered'
                    agent.symptomatic = False
                    agent.hospitalized = False
                    agent.inIntensiveCare = False
                    agent.workingShare = 1.0
                    agent.mildConditionIndex = 0
                    agent.contactReductionRate = 1.0
                    agent.viralLoad = 0
                noMoreInfectious += 1
      
        
#        totalHospitalized = self.hospitalized1To11+self.hospitalized12To18+self.hospitalized19To39+self.hospitalized40To64+self.hospitalizedOver65
#        totIntubated = self.icu1To11+self.icu12To18+self.icu19To39+self.icu40To64+self.icuOver65
#        print 'Total hospitalized: ' + str(totalHospitalized)
#        print 'Total intubated: ' + str(totIntubated)
#        print 'Total infected: ' + str(self.totalInfected)
#        if self.totalInfected > 0:
#            print ''
#            print 'Share of 1/11-year-old infected: ' + str(100*float(self.infected1To11)/float(self.totalInfected)) + ' (5.5)'
#            print 'Share of 12/18-year-old infected: ' + str(100*float(self.infected12To18)/float(self.totalInfected)) + ' (7.2)'
#            print 'Share of 19/39-year-old infected: ' + str(100*float(self.infected19To39)/float(self.totalInfected)) + ' (31.0)'
#            print 'Share of 40/64-year-old infected: ' + str(100*float(self.infected40To64)/float(self.totalInfected)) + ' (39.4)'
#            print 'Share of over 65 infected: ' + str(100*float(self.infectedOver65)/float(self.totalInfected)) + ' (16.7)'
#            print 'Ratio hospitalized/infected: ' + str(100*float(totalHospitalized)/float(self.totalInfected)) + ' (6.71)'
          
#        self.Age1To11Share_H = 0
#        self.Age12To18Share_H = 0
#        self.Age19To39Share_H = 0
#        self.Age40To64Share_H = 0
#        self.Over65Share_H = 0
#        if totalHospitalized > 0:
#            self.Age1To11Share_H = 100*float(self.hospitalized1To11)/float(totalHospitalized)
#            self.Age12To18Share_H = 100*float(self.hospitalized12To18)/float(totalHospitalized)
#            self.Age19To39Share_H = 100*float(self.hospitalized19To39)/float(totalHospitalized)
#            self.Age40To64Share_H = 100*float(self.hospitalized40To64)/float(totalHospitalized)
#            self.Over65Share_H = 100*float(self.hospitalizedOver65)/float(totalHospitalized)
#            
#        print 'Share of 1/11-year-old hospitalized: ' + str(self.Age1To11Share_H) + ' (0.8)'
#        print 'Share of 12/18-year-old hospitalized: ' + str(self.Age12To18Share_H) + ' (0.9)'
#        print 'Share of 19/39-year-old hospitalized: ' + str(self.Age19To39Share_H) + ' (9.0)'
#        print 'Share of 40/64-year-old hospitalized: ' + str(self.Age40To64Share_H) + ' (31.5)'
#        print 'Share of over 65 hospitalized: ' + str(self.Over65Share_H) + ' (57.9)'
#        print ''
#        
#        self.Age1To11Share_I = 0
#        self.Age12To18Share_I = 0
#        self.Age19To39Share_I = 0
#        self.Age40To64Share_I = 0
#        self.Over65Share_I = 0
#        if totIntubated > 0:
#            self.Age1To11Share_I = 100*float(self.icu1To11)/float(totIntubated)
#            self.Age12To18Share_I = 100*float(self.icu12To18)/float(totIntubated)
#            self.Age19To39Share_I = 100*float(self.icu19To39)/float(totIntubated)
#            self.Age40To64Share_I = 100*float(self.icu40To64)/float(totIntubated)
#            self.Over65Share_I = 100*float(self.icuOver65)/float(totIntubated)
#            
#        print 'Share of 1/11-year-old intubated: ' + str(self.Age1To11Share_I) + ' (0.2)'
#        print 'Share of 12/18-year-old intubated: ' + str(self.Age12To18Share_I) + ' (0.3)'
#        print 'Share of 19/39-year-old intubated: ' + str(self.Age19To39Share_I) + ' (3.4)'
#        print 'Share of 40/64-year-old intubated: ' + str(self.Age40To64Share_I) + ' (31.0)'
#        print 'Share of over 65 intubated: ' + str(self.Over65Share_I) + ' (65.0)'
#        print ''
    
        totDeaths = sum(self.deathsByAge)
        
        totIntubated = sum(self.intubatedByAge)
        totHospitalized = sum(self.hospitalizedByAge)
        totMild = sum(self.symptomaticByAge)
        totAsymptomatic = sum(self.infectedByAge) - (totMild+totHospitalized+totIntubated+totDeaths)
        numConditions = [totAsymptomatic, totMild, totHospitalized, totIntubated, totDeaths]
        self.shareConditions = [0, 0, 0, 0, 0]
        if sum(numConditions) > 0:
            self.shareConditions = [float(x)/float(sum(numConditions)) for x in numConditions]
        
        self.pop.livingPeople[:] = [x for x in self.pop.livingPeople if x.dead == False]
        postPop = len(self.pop.livingPeople)
        self.deathsForCovid = prePop-postPop
        self.totalDeaths += self.deathsForCovid
        
#        if self.totalDeaths > 0:
#            print ''
#            print 'Share of 1/11-year-old dead: ' + str(100*float(self.dead1To11)/float(self.totalDeaths)) + ' (0.0)'
#            print 'Share of 12/18-year-old dead: ' + str(100*float(self.dead12To18)/float(self.totalDeaths)) + ' (0.1)'
#            print 'Share of 19/39-year-old dead: ' + str(100*float(self.dead19To39)/float(self.totalDeaths)) + ' (0.6)'
#            print 'Share of 40/64-year-old dead: ' + str(100*float(self.dead40To64)/float(self.totalDeaths)) + ' (8.1)'
#            print 'Share of over 65 dead: ' + str(100*float(self.deadOver65)/float(self.totalDeaths)) + ' (91.2)'
#            print ''
        
        self.deathsToInfectedRatio = 0
        if self.totalInfected > 0:
            self.deathsToInfectedRatio = float(self.totalDeaths)/float(self.totalInfected)
            
        print 'Total deaths: ' + str(self.totalDeaths)
        print 'Total infected: ' + str(self.totalInfected)
        
        self.deathsToInfectedRatios.append(self.deathsToInfectedRatio)
        print 'New cases : ' + str(self.newCases)
        
        if self.lockdown == False and self.p['lockdownEvent'] == 'death' and self.periodFirstDeath != -1 and self.lockdownDay == -1:
            self.lockdownDay = day + self.p['daysFromEvent']
        
#        self.R0 = 0
#        if len(susceptibles) > 0 and len(infectiousPeople) > 0:
#            beta = (float(newExposed)/float(len(susceptibles)))*(float(self.p['population'])/(float(len(infectiousPeople))))
#            gamma = float(noMoreInfectious)/float(len(infectiousPeople))
#            if gamma != 0:
#                self.R0 = beta/gamma
            
    def resetEventsIndicators(self, house):
        house.experiencedEvent = False
        house.kindOfEvent = None
        house.infectionEvent = False
        house.contactReductionRate = -1
        house.isolationFactor = -1
        house.individualIsolation = -1
        house.neighborIsolation = -1
        
    def updatePublicInfo(self):
        
        self.newCasesRatio = float(self.newCases)/float(len(self.pop.livingPeople))
        # For the public information salience
        self.relativeCasesIncrease = float(self.newCases-self.pastCases)/float(len(self.pop.livingPeople))
        self.newCasesRatios.append(self.newCasesRatio)
        self.probsConditionsByAge[:] = []
        self.varsConditionsByAge[:] = []
        
        newCasesDiscounted = 0
        den = 0
        for i in range(len(self.newCasesRatios)):
            index = len(self.newCasesRatios)-1
            newCasesDiscounted += self.newCasesRatios[index-i]*np.power(self.p['timeDiscountingFactor'], i)
            den += np.power(self.p['timeDiscountingFactor'], i)
        self.newCasesIndex = float(newCasesDiscounted)/float(den)
        
        print 'New case index: ' + str(self.newCasesIndex)
        # print 'Share susceptible: ' + str(shareSusceptible)
        
        self.shareHospitalized1To18 = 0 
        self.shareIntubated1To18 = 0
        self.shareDeaths1To18 = 0
        if self.infected1To18 > 0:
            self.shareHospitalized1To18 = float(self.hospitalized1To18)/float(self.infected1To18)
            self.shareIntubated1To18 = float(self.icu1To18)/float(self.infected1To18)
            self.shareDeaths1To18 = float(self.dead1To18)/float(self.infected1To18)
        
        self.shareHospitalized1To18_TS.append(self.shareHospitalized1To18) 
        self.shareIntubated1To18_TS.append(self.shareIntubated1To18)
        self.shareDeaths1To18_TS.append(self.shareDeaths1To18)
        
#        sampleSize = min(len(self.shareHospitalized1To18_TS), int(self.p['maxSampleSize']))
#        self.shareHospitalized1To18_var = np.var(self.shareHospitalized1To18_TS[-sampleSize:])
#        self.shareIntubated1To18_var = np.var(self.shareIntubated1To18_TS[-sampleSize:])
#        self.shareDeaths1To18_var = np.var(self.shareDeaths1To18_TS[-sampleSize:])
        
        probCond1To18 = [self.shareHospitalized1To18, self.shareIntubated1To18, self.shareDeaths1To18]
        # varCond1To18 = [self.shareHospitalized1To18_var, self.shareIntubated1To18_var, self.shareDeaths1To18_var]
        
        self.shareHospitalized19To39 = 0 
        self.shareIntubated19To39 = 0
        self.shareDeaths19To39 = 0
        if self.infected19To39 > 0:
            self.shareHospitalized19To39 = float(self.hospitalized19To39)/float(self.infected19To39)
            self.shareIntubated19To39 = float(self.icu19To39)/float(self.infected19To39)
            self.shareDeaths19To39 = float(self.dead19To39)/float(self.infected19To39)
            
        self.shareHospitalized19To39_TS.append(self.shareHospitalized19To39) 
        self.shareIntubated19To39_TS.append(self.shareIntubated19To39)
        self.shareDeaths19To39_TS.append(self.shareDeaths19To39)
        
#        self.shareHospitalized19To39_var = np.var(self.shareHospitalized19To39_TS[-sampleSize:]) 
#        self.shareIntubated19To39_var = np.var(self.shareIntubated19To39_TS[-sampleSize:])
#        self.shareDeaths19To39_var = np.var(self.shareDeaths19To39_TS[-sampleSize:])
        
        probCond19To39 = [self.shareHospitalized19To39, self.shareIntubated19To39, self.shareDeaths19To39]
        # varCond19To39 = [self.shareHospitalized19To39_var, self.shareIntubated19To39_var, self.shareDeaths19To39_var]
        
        self.shareHospitalized40To59 = 0 
        self.shareIntubated40To59 = 0
        self.shareDeaths40To59 = 0
        if self.infected40To59 > 0:
            self.shareHospitalized40To59 = float(self.hospitalized40To59)/float(self.infected40To59)
            self.shareIntubated40To59 = float(self.icu40To59)/float(self.infected40To59)
            self.shareDeaths40To59 = float(self.dead40To59)/float(self.infected40To59)
            
        self.shareHospitalized40To59_TS.append(self.shareHospitalized40To59) 
        self.shareIntubated40To59_TS.append(self.shareIntubated40To59)
        self.shareDeaths40To59_TS.append(self.shareDeaths40To59)    
        
#        self.shareHospitalized40To59_var = np.var(self.shareHospitalized19To39_TS[-sampleSize:]) 
#        self.shareIntubated40To59_var = np.var(self.shareIntubated19To39_TS[-sampleSize:])
#        self.shareDeaths40To59_var = np.var(self.shareDeaths19To39_TS[-sampleSize:])
        
        probCond40To59 = [self.shareHospitalized40To59, self.shareIntubated40To59, self.shareDeaths40To59]
        # varCond40To59 = [self.shareHospitalized40To59_var, self.shareIntubated40To59_var, self.shareDeaths40To59_var]
        
        self.shareHospitalized60To79 = 0 
        self.shareIntubated60To79 = 0
        self.shareDeaths60To79 = 0
        if self.infected60To79 > 0:
            self.shareHospitalized60To79 = float(self.hospitalized60To79)/float(self.infected60To79)
            self.shareIntubated60To79 = float(self.icu60To79)/float(self.infected60To79)
            self.shareDeaths60To79 = float(self.dead60To79)/float(self.infected60To79)
        
        self.shareHospitalized60To79_TS.append(self.shareHospitalized60To79) 
        self.shareIntubated60To79_TS.append(self.shareIntubated60To79)
        self.shareDeaths60To79_TS.append(self.shareDeaths60To79)
        
#        self.shareHospitalized60To79_var = np.var(self.shareHospitalized60To79_TS[-sampleSize:]) 
#        self.shareIntubated60To79_var = np.var(self.shareIntubated60To79_TS[-sampleSize:])
#        self.shareDeaths60To79_var = np.var(self.shareDeaths60To79_TS[-sampleSize:])
        
        probCond60To79 = [self.shareHospitalized60To79, self.shareIntubated60To79, self.shareDeaths60To79]
        # varCond60To79 = [self.shareHospitalized60To79_var, self.shareIntubated60To79_var, self.shareDeaths60To79_var]
            
        self.shareHospitalized80Over = 0 
        self.shareIntubated80Over = 0
        self.shareDeaths80Over = 0
        if self.infected80Over > 0:
            self.shareHospitalized80Over = float(self.hospitalized80Over)/float(self.infected80Over)
            self.shareIntubated80Over = float(self.icu80Over)/float(self.infected80Over)
            self.shareDeaths80Over = float(self.dead80Over)/float(self.infected80Over)
        
        self.shareHospitalized80Over_TS.append(self.shareHospitalized80Over) 
        self.shareIntubated80Over_TS.append(self.shareIntubated80Over)
        self.shareDeaths80Over_TS.append(self.shareDeaths80Over)
        
#        self.shareHospitalized80Over_var = np.var(self.shareHospitalized80Over_TS[-sampleSize:]) 
#        self.shareIntubated80Over_var = np.var(self.shareIntubated80Over_TS[-sampleSize:])
#        self.shareDeaths80Over_var = np.var(self.shareDeaths80Over_TS[-sampleSize:])
        
        probCond80Over = [self.shareHospitalized80Over, self.shareIntubated80Over, self.shareDeaths80Over]
        # varCond80Over = [self.shareHospitalized80Over_var, self.shareIntubated80Over_var, self.shareDeaths80Over_var]
        
        # It stores the probabilities to be hospitalized, intubated or to die if infected, according to age class
        self.probsConditionsByAge = [probCond1To18, probCond19To39, probCond40To59, probCond60To79, probCond80Over]
        # self.varsConditionsByAge = [varCond1To18, varCond19To39, varCond40To59, varCond60To79, varCond80Over]
        
        for town in self.map.towns:
            townPop = len([x for x in self.pop.livingPeople if x.house.town == town])
            if townPop > 0:
                town.casesByDay.append(town.newCases)
                town.newCasesRatio = 0
                if townPop > 0:
                    town.newCasesRatio = float(town.newCases)/float(townPop)
                town.newCasesRatios.append(town.newCasesRatio)
                town.relativeCasesIncrease = float(town.newCases-town.pastCases)/float(townPop)
                newCasesDiscounted = 0
                den = 0
                for i in range(len(town.newCasesRatios)):
                    index = len(town.newCasesRatios)-1
                    newCasesDiscounted += town.newCasesRatios[index-i]*np.power(self.p['timeDiscountingFactor'], i)
                    den += np.power(self.p['timeDiscountingFactor'], i)
                town.discountedCasesRatio = 0
                if den > 0:
                    town.discountedCasesRatio = float(newCasesDiscounted)/float(den)
        
        
#        print 'Share of 1/11-year-old hospitalized: ' + str(self.shareHospitalized1To11) + ' (2.5)'
#        print 'Share of 12/18-year-old hospitalized: ' + str(self.shareHospitalized12To18) + ' (1.5)'
#        print 'Share of 19/39-year-old hospitalized: ' + str(self.shareHospitalized19To39) + ' (1.9)'
#        print 'Share of 40/64-year-old hospitalized: ' + str(self.shareHospitalized40To64) + ' (5.4)'
#        print 'Share of over 65 hospitalized: ' + str(self.shareHospitalizedOver65) + ' (23.3)'
#        
#        print 'Share of 1/11-year-old intubated: ' + str(self.shareIntubated1To11) + ' (0.09)'
#        print 'Share of 12/18-year-old intubated: ' + str(self.shareIntubated12To18) + ' (0.06)'
#        print 'Share of 19/39-year-old intubated: ' + str(self.shareIntubated19To39) + ' (0.09)'
#        print 'Share of 40/64-year-old intubated: ' + str(self.shareIntubated40To64) + ' (0.67)'
#        print 'Share of over 65 intubated: ' + str(self.shareIntubatedOver65) + ' (3.32)'
#        print ''
#        print 'Share of 1/11-year-old dead: ' + str(self.shareDeaths1To11) + ' (0.01)'
#        print 'Share of 12/18-year-old dead: ' + str(self.shareDeaths12To18) + ' (0.02)'
#        print 'Share of 19/39-year-old dead: ' + str(self.shareDeaths19To39) + ' (0.02)'
#        print 'Share of 40/64-year-old dead: ' + str(self.shareDeaths40To64) + ' (0.24)'
#        print 'Share of over 65 dead: ' + str(self.shareDeathsOver65) + ' (6.31)
                
    def computeCostOfInfection(self):
        # Here, the cost of being infecetd is computed
        # It depends on the intrinsic cost of a condition (hospitalized, intubated, or death) and 
        # the subective probability of this condition
        probsHospitalPreNet = []
        probsICUPreNet = []
        probsDeathPreNet = []
        
        probsHospitalPostNet = []
        probsICUPostNet = []
        probsDeathPostNet = []
        
        hpFactors = []
        vpFactors = []
        dpFactors = []
        
        popByAgeClasses = [len([y for y in self.pop.livingPeople if y.ageClass == x]) for x in range(int(self.p['ageClasses']))]
        for house in self.map.occupiedHouses:
            for agent in house.occupants:
                # - Start from public (empirical) probabilities of hospitalization, ventilation and death
                # - The empirical probabilities act as attraction points on the actual probabilities
                ageGroup = int(min(int(agent.age/10), self.p['ageClasses']-1))
                # Expected hospitalization cost
                probH = self.p['probSymptomatic'][ageGroup]*self.p['probsHospitalization'][ageGroup]
                
                hospitalizationsRatio = 0
                if popByAgeClasses[ageGroup] > 0:
                    hospitalizationsRatio = float(self.hospitalizedByAge[ageGroup])/float(popByAgeClasses[ageGroup])
                rateToProb = probH+(1.0-probH)*(1.0-1.0/np.exp(self.p['rateToProbCond']*hospitalizationsRatio))
                deltaProb = rateToProb-agent.probHospital
                agent.probHospital += deltaProb*self.p['probAdjustRate']
                
                probsHospitalPreNet.append(agent.probHospital)
                
                agent.probHospital = self.socialLearning_Cond(agent, agent.probHospital, 'hospitalized')
            
                probsHospitalPostNet.append(agent.probHospital)
            
                hpFactor = (1.0-agent.probHospital)/agent.probHospital
                
                hpFactors.append(hpFactor)
                
                den = np.power(1.0+self.p['condProbFactor']*hpFactor, self.p['condProbExp'])
                cost = self.p['conditionCostFactor']
                costH = cost/den
                
                # Expected intubation cost
                probV = probH*self.p['probsIntensiveCare'][ageGroup]
               
                icusRatio = 0
                if popByAgeClasses[ageGroup] > 0:
                    icusRatio = float(self.intubatedByAge[ageGroup])/float(popByAgeClasses[ageGroup])
                rateToProb = probV+(1.0-probV)*(1.0-1.0/np.exp(self.p['rateToProbCond']*icusRatio))
                deltaProb = rateToProb-agent.probVentilation
                agent.probVentilation += deltaProb*self.p['probAdjustRate']
                
                probsICUPreNet.append(agent.probVentilation)
                
                agent.probVentilation = self.socialLearning_Cond(agent, agent.probVentilation, 'intubated')
                
                probsICUPostNet.append(agent.probVentilation)
                
                vpFactor = (1.0-agent.probVentilation)/agent.probVentilation
                
                vpFactors.append(vpFactor)
                
                den = np.power(1.0+self.p['condProbFactor']*vpFactor, self.p['condProbExp'])
                cost *= self.p['conditionCostMultiplier']
                costV = cost/den
                
                # Expected death cost

                probD = self.ifr[ageGroup]/100
                deathsRatio = 0
                if popByAgeClasses[ageGroup] > 0:
                    deathsRatio = float(self.deathsByAge[ageGroup])/float(popByAgeClasses[ageGroup])
                rateToProb = probD+(1.0-probD)*(1.0-np.exp(-1*self.p['rateToProbCond']*deathsRatio))
                deltaProb = rateToProb-agent.probDeath
                agent.probDeath += deltaProb*self.p['probAdjustRate']
                
                probsDeathPreNet.append(agent.probDeath)
                
                agent.probDeath = self.socialLearning_Cond(agent, agent.probDeath, 'death')
                
                probsDeathPostNet.append(agent.probDeath)
                
                dpFactor = (1.0-agent.probDeath)/agent.probDeath
                
                dpFactors.append(dpFactor)
                
                den = np.power(1.0+self.p['condProbFactor']*dpFactor, self.p['condProbExp'])
                cost *= self.p['conditionCostMultiplier']
                costD = cost/den
                
                agent.costInfection = costH+costV+costD
                
        print ''
        
        print 'Mean prob hospitalization (pre): ' + str(np.mean(probsHospitalPreNet))
        print 'Median prob hospitalization (pre): ' + str(np.median(probsHospitalPreNet))
        print 'Mean prob hospitalization (post): ' + str(np.mean(probsHospitalPostNet))
        print 'Median prob hospitalization (post): ' + str(np.median(probsHospitalPostNet))
        print 'Mean hp factors: ' + str(np.mean(hpFactors))
        print 'Median hp factors: ' + str(np.median(hpFactors))
        
        print 'Mean prob ventilation (pre): ' + str(np.mean(probsICUPreNet))
        print 'Median prob ventilation (pre): ' + str(np.median(probsICUPreNet))
        print 'Mean prob ventilation (post): ' + str(np.mean(probsICUPostNet))
        print 'Median prob ventilation (post): ' + str(np.median(probsICUPostNet))
        print 'Mean vp factors: ' + str(np.mean(vpFactors))
        print 'Median vp factors: ' + str(np.median(vpFactors))
        
        print 'Mean prob death (pre): ' + str(np.mean(probsDeathPreNet))
        print 'Median prob death (pre): ' + str(np.median(probsDeathPreNet))
        print 'Mean prob death (post): ' + str(np.mean(probsDeathPostNet))
        print 'Median prob death (post): ' + str(np.median(probsDeathPostNet))
        print 'Mean dp factors: ' + str(np.mean(dpFactors))
        print 'Median dp factors: ' + str(np.median(dpFactors))
        
        print ''
    
    
    def performTest(self):
        
        # Testing process
        adults = [x for x in self.pop.livingPeople if x.age > 15]
        testAgents = [x for x in adults if x.symptomsLevel == 'none' or x.symptomsLevel == 'asymptomatic' or x.symptomsLevel == 'mild' or x.placeOfDeath == 'Home']
        self.periodTests = 0
        allExpectedCosts = []
        allInfectedIndexes = []
        allSensitivityToRisks = []
        riskIndexes = []
        riskProbs = []
        for person in testAgents:
            person.sensitivityToRisk = 0
            probTest = 0
            if person.testPositive == True:
                person.daysSincePositive += 1
                if person.daysSincePositive == self.p['infectionMeanLength']:
                    probTest = 1.0
            else:
                # In this case, the agent has not taken the test or it take it but was negative
                person.daysSinceLastTest += 1
                
                # First element of probability of test: how is it likely that the person got the virus?
                ageGroup = int(min(int(person.age/10), self.p['ageClasses']-1))
                probAsymptomatic = 1.0-self.p['probSymptomatic'][ageGroup]
                symptomsIndex = max(math.pow(person.mildConditionIndex, self.p['symptomsTestExp']), probAsymptomatic)
                
                # Function of numer of infected agents in the household
                houseInfections = len([x for x in person.house.inHouseEvents if x.condition == 'infected'])
                householdInfectionIndex = 1.0-np.exp(-1*self.p['houseInfectionsTestBeta']*houseInfections)
                riskIndex = 0
                maxLen = min(self.p['infectionMeanLength'], len(person.isolationRates))
                for i in range(int(maxLen)):
                    ti = i+1
                    riskIndex += (1.0-person.isolationRates[-ti])*person.probInfections[-ti]*np.power(self.p['timeDiscountingFactor'], i)
                riskProb = 1.0-np.exp(-1*self.p['riskTestBeta']*riskIndex)
                
                riskProbs.append(riskProb)
                
                person.infectedIndex = (riskProb+householdInfectionIndex)*symptomsIndex
                
                allInfectedIndexes.append(person.infectedIndex)
                riskIndexes.append(riskIndex)
                
                if person.infectedIndex > 0:
                    ipFactor = (1.0-person.infectedIndex)/person.infectedIndex
                    den = np.power(1.0+self.p['infectionProbFactor']*ipFactor, self.p['infectionProbExp'])
                    expectedTotalCost = person.costInfection/den 
                    
                    allExpectedCosts.append(expectedTotalCost)
                    
                    incomeFactor = (person.incomeQuintile+1.0)/5.0
                    person.sensitivityToRisk = self.p['testProbFactor']*person.socialPreferenceIndex*incomeFactor
                    
                    probTest = 1.0 - np.exp(-1*person.sensitivityToRisk*expectedTotalCost)
                else:
                    probTest = 0.0
                
            allSensitivityToRisks.append(person.sensitivityToRisk)
            
            if np.random.random() < probTest:
                self.periodTests += 1
                person.testPositive = False
                person.daysSinceLastTest = 0
                person.infectedIndex = 0
                # If the person is infectious, the test may be positive
                if (person.healthStatus == 'infectious' and np.random.random() > self.p['probFalseNegative']) or (person.healthStatus != 'infectious' and np.random.random() < self.p['probFalsePositive']):
                    person.testPositive = True
                    person.hadPandemicEvent = True
                    person.daysSincePositive = 0
                    newEvent = PandemicEvent(person, person.age, 'infected', person.isolationRate, 0)
                    person.house.inHouseEvents.append(newEvent)
                    neighbors = [x for x in person.house.infoNetwork.neighbors(person.house)]
                    for neighbor in neighbors:
                        weight = neighbor.infoNetwork[neighbor][person.house]['weight']
                        newEvent.distanceWeight = weight
                        neighbor.periodEvents.append(newEvent)
                    
                    # With a certain probability, the positive result is recorded by the public
                    if np.random.random() < self.p['testResultsDiffusion']:
                        self.newCases += 1
                        person.house.town.newCases += 1
                        if person.age < 19:
                            self.infected1To18 += 1
                        if person.age >= 19 and person.age < 40:
                            self.infected19To39 += 1
                        if person.age >= 40 and person.age < 59:
                            self.infected40To59 += 1
                        if person.age >= 60 and person.age < 79:
                            self.infected60To79 += 1
                        if person.age >= 80:
                            self.infected80Over += 1
                else:
                    person.testPositive = False
        
        shareTested = 0
        if len(testAgents) > 0:
            shareTested = float(self.periodTests)/float(len(testAgents))
        print 'Share testes: ' + str(shareTested)
        
        if len(riskProbs) > 0:
            print 'Max risk probs: ' + str(max(riskProbs))
            print 'Mean risk probs: ' + str(np.mean(riskProbs))
            print 'Median risk probs: ' + str(np.median(riskProbs))
        
        if len(riskIndexes) > 0:
            print 'Max risk index: ' + str(max(riskIndexes))
            print 'Mean risk index: ' + str(np.mean(riskIndexes))
            print 'Median risk index: ' + str(np.median(riskIndexes))
        
        if len(allInfectedIndexes) > 0:
            print 'Max infected index: ' + str(max(allInfectedIndexes))
            print 'Mean infected index: ' + str(np.mean(allInfectedIndexes))
            print 'Median infected index: ' + str(np.median(allInfectedIndexes))
        
        if len(allSensitivityToRisks) > 0:
            print 'Max sensitivity to risk: ' + str(max(allSensitivityToRisks))
            print 'Mean sensitivity to risk: ' + str(np.mean(allSensitivityToRisks))
            print 'Median sensitivity to risk: ' + str(np.median(allSensitivityToRisks))
        
        if len(allExpectedCosts) > 0:
            print 'Max expected cost (test): ' + str(max(allExpectedCosts))   
            print 'Median expected cost (test): ' + str(np.median(allExpectedCosts))
            
    def updateInfectionRisk(self):
        # Here, the probability fo infection is updated, for each agent
        
        probInfectionPre = []
        probInfectionPost = []
        newCasesRatios = []
        for house in self.map.occupiedHouses:
            adults = [x for x in house.occupants if x.age > 15]
            for person in adults:
                # Number of new cases (public info)
                rateToProb = 1.0-np.exp(-1*self.p['rateToProbExp']*house.town.newCasesRatio)
                deltaCaseRatio = rateToProb-person.probInfection
                person.probInfection += deltaCaseRatio*self.p['probAdjustRate']
                
                # Debugging code
                probInfectionPre.append(person.probInfection)
                newCasesRatios.append(house.town.newCasesRatio)
                
                # Social interaction
                events = [x for x in house.periodEvents if x.condition == 'infected']
                houseEvents = [x for x in house.inHouseEvents if x.condition == 'infected']
                friendsWithEvents = [x.agent for x in events]
#                friendsWithNoEvents = []
#                neighbors = [x for x in house.infoNetwork.neighbors(house)]
#                for neighbor in neighbors:
#                    neighborMembers = [x for x in neighbor.occupants]
#                    # Each event changes the perceived risk
#                    # The observation of friends without a pandemic event also changes the perception of risk
#                    # friends = [x for x in friendsWithEvents]
#                    for friend in neighborMembers:
#                        # Check if friend had a pandemic event
#                        if friend.hadPandemicEvent == False:
#                            friendsWithNoEvents.append(friend)
                membersWithEvents = [x.agent for x in houseEvents if x != person]
                # membersWithNoEvents = [x for x in house.occupants if x.hadPandemicEvent == False and x != person]
                allNeighbors = friendsWithEvents+membersWithEvents# +membersWithNoEvents+friendsWithNoEvents
                if len(allNeighbors) > 0:
                    np.random.shuffle(allNeighbors)
                    for agent in allNeighbors:
                        # Implement change according to observation of friend's events
                        # if agent.hadPandemicEvent == True:
                        distance = 0
                        if agent in friendsWithEvents:
                            singleEvent = [x for x in events if x.agent == agent][0]
                            distance = singleEvent.distanceWeight
                            # Events of households member have the highest effect on perceived 
                        weight = np.exp(self.p['riskDistSensitivity']*distance)
                        deltaIsolation = agent.isolationRate-person.isolationRate
                        # The event will have the greater effect the more the infected person was younger
                        expRate = np.exp(self.p['deltaIRExp']*deltaIsolation)/(np.exp(self.p['deltaIRExp']*deltaIsolation)+1.0)
                        person.probInfection += (1.0-person.probInfection)*expRate/weight
                else: # In this case, the probability is reduced
                    person.probInfection -= person.probInfection*(1.0-np.exp(-1*self.p['fearExtintionRate'])) #expRate/weight
                    
                    
#                    distance = 0
#                    if agent in friendsWithNoEvents:
#                        distance = house.infoNetwork[house][agent.house]['weight']
#                    weight = np.exp(self.p['riskDistSensitivity']*distance)
#                    deltaIsolation = person.isolationRate-agent.isolationRate
#                    expRate = np.exp(self.p['deltaIRExp']*deltaIsolation)/(np.exp(self.p['deltaIRExp']*deltaIsolation)+1.0)
                    # person.probInfection -= person.probInfection*expRate/weight
                probInfectionPost.append(person.probInfection)
                
                person.probInfections.append(person.probInfection)
                
        print 'Mean prob infection (pre): ' + str(np.mean(probInfectionPre))
        print 'Median prob infection (pre): ' + str(np.median(probInfectionPre))
        
        print 'Mean cases ratio: ' + str(np.mean(newCasesRatios))
        print 'Median cases ratio: ' + str(np.median(newCasesRatios))
        
        print 'Mean prob infection (post): ' + str(np.mean(probInfectionPost))
        print 'Median prob infection (post): ' + str(np.median(probInfectionPost))
        
    
    def showProbs(self):
        for i in range(int(self.p['ageClasses'])):
            for j in range(int(self.p['incomeClasses'])):
                for z in range(2):
                    print 'Age: ' + str(i) + 'Income: ' + str(j) + 'Gender: ' + str(z) + 'Prob Symptomatic: ' + str(self.probSymptomatic[i][j][z])
                    print 'Age: ' + str(i) + 'Income: ' + str(j) + 'Gender: ' + str(z) + 'Prob Hospital: ' + str(self.probsHospitalization[i][j][z])
                    print 'Age: ' + str(i) + 'Income: ' + str(j) + 'Gender: ' + str(z) + 'Prob ICU: ' + str(self.probsIntensiveCare[i][j][z])
                    print 'Age: ' + str(i) + 'Income: ' + str(j) + 'Gender: ' + str(z) + 'Prob Death: ' + str(self.intubatedFatalityRatio[i][j][z])
                    print 'Age: ' + str(i) + 'Income: ' + str(j) + 'Gender: ' + str(z) + 'IFR: ' + str(self.infectionFatalityRatio[i][j][z])
                    
        # print 'Did doStats'
    def setPandemicWeights(self):
        
#        for house in self.map.occupiedHouses:
#            incomeQuintile = max([x.incomeQuintile for x in house.occupants])
#            householdRank = max([x.classRank for x in house.occupants])
#            house.strategyFactor = self.p['weightIncome']*float(incomeQuintile)+(1.0-self.p['weightIncome'])*float(householdRank)
#            house.publicInfoWeight = 1.0/np.exp(self.p['publicInfoBeta']*house.strategyFactor)
        
        sizeAgeGroups = []
        for i in range(int(self.p['numMortalityAgeClasses']+1)):
            sizeAgeGroups.append(len([x for x in self.pop.livingPeople if x.mortalityAgeClass == i]))
        
        self.ifr = []
        for i in range(0, int(self.p['numMortalityAgeClasses'])-1, 2):
            if i < 16:
                f1 = float(sizeAgeGroups[i])/float(sizeAgeGroups[i]+sizeAgeGroups[i+1])
                f2 = float(sizeAgeGroups[i+1])/float(sizeAgeGroups[i]+sizeAgeGroups[i+1])
                self.ifr.append(self.p['infectionFatalityByAge'][i]*f1+self.p['infectionFatalityByAge'][i+1]*f2)
            else:
                f1 = float(sizeAgeGroups[i])/float(sizeAgeGroups[i]+sizeAgeGroups[i+1]+sizeAgeGroups[i+2])
                f2 = float(sizeAgeGroups[i+1])/float(sizeAgeGroups[i]+sizeAgeGroups[i+1]+sizeAgeGroups[i+2])
                f3 = float(sizeAgeGroups[i+2])/float(sizeAgeGroups[i]+sizeAgeGroups[i+1]+sizeAgeGroups[i+2])
                self.ifr.append(self.p['infectionFatalityByAge'][i]*f1+self.p['infectionFatalityByAge'][i+1]*f2+self.p['infectionFatalityByAge'][i+2]*f3)
        
        self.probsDeathIntubated = []
        for i in range(int(self.p['ageClasses'])):
            infectionIcuRate = self.p['probSymptomatic'][i]*self.p['probsHospitalization'][i]*self.p['probsIntensiveCare'][i]
            
            print 'IFR: ' + str(self.ifr[i])
            print 'Infection ICU rate: ' + str(infectionIcuRate)
            
            # self.probsDeathIntubated.append((self.ifr[i]/100)/infectionIcuRate)
            
            ## Using the Ferguson et all paper Report 9 data.
            self.probsDeathIntubated.append((self.p['infectionFatalityRatio'][i]/100.0)/infectionIcuRate)
        
        print 'Probs deaths if intubated: ' + str(self.probsDeathIntubated)
        
        # Income- and gender-specific probabilities
        
        # Share of population in each age group by income class
        sharesQuintilesByAge = []
        sharesMalesByAgeAndIncome = []
        for i in range(int(self.p['ageClasses'])):
            sharesQuintile = []
            sharesMales = []
            if i < self.p['ageClasses']-1:
                popAge = [x for x in self.pop.livingPeople if x.age >= i*self.p['ageRange'] and x.age < (i+1)*self.p['ageRange']-1]
            else:
                popAge = [x for x in self.pop.livingPeople if x.age >= i*self.p['ageRange']]
            for j in range(int(self.p['incomeClasses'])):
                popClass = [x for x in popAge if x.incomeQuintile == j]
                popMaleInClass = [x for x in popClass if x.sex == 'male']
                shareGender = 0
                if len(popClass) > 0:
                    shareGender = float(len(popMaleInClass))/float(len(popClass))
                sharesMales.append(shareGender)
                shareQuintile = 0
                if len(popAge) > 0:
                    shareQuintile = float(len(popClass))/float(len(popAge))
                sharesQuintile.append(shareQuintile)
            sharesQuintilesByAge.append(sharesQuintile)
            sharesMalesByAgeAndIncome.append(sharesMales)
        
        # Share of males (and females) by age and income group
        
#        for i in range(int(self.p['ageClasses'])):
#            if i < self.p['ageClasses']-1:
#                popAge = [x for x in self.pop.livingPeople if x.age > i*self.p['ageRange'] and x.age < (i+1)*self.p['ageRange']-1]
#            else:
#                popAge = [x for x in self.pop.livingPeople if x.age > i*self.p['ageRange']]
#            maleAge = [x for x in popAge if x.sex == 'male']
#            sharesMalesByAge.append(float(len(maleAge))/float(len(popAge)))
        
        # den = (1.0/float(self.p['incomeClasses']))*a
        for i in range(int(self.p['ageClasses'])):
            
            den = 0
            for j in range(int(self.p['incomeClasses'])):
                den += sharesQuintilesByAge[i][j]*math.pow(self.p['severityClassBias'], j)
            
            probClassAgeS = []
            probClassAgeH = []
            probClassAgeI = []
            probClassAgeF = []
            probClassAgeR = []
            baseProbS = self.p['probSymptomatic'][i]/den
            
            # baseProbH = self.p['probsHospitalization'][i]/den
            probHosp = 1.0 - np.exp(-1*self.p['betaHosp']*np.power((i+1), self.p['alphaHosp']))
            baseProbH = probHosp/den
            
            # baseProbI = self.p['probsIntensiveCare'][i]/den
            probICU = 1.0 - np.exp(-1*self.p['betaICU']*np.power((i+1), self.p['alphaICU']))
            baseProbI = probICU/den
            
            baseProbF = self.probsDeathIntubated[i]/den
            
            # baseProbR = (self.p['infectionFatalityRatio'][i]/100.0)/den
            probR = 1.0 - np.exp(-1*self.p['betaIFR']*np.power((i+1), self.p['alphaIFR']))
            baseProbR = probR/den
            
            for j in range(int(self.p['incomeClasses'])):
                classProbS = baseProbS*math.pow(self.p['severityClassBias'], j)
                classProbH = baseProbH*math.pow(self.p['severityClassBias'], j)
                classProbI = baseProbI*math.pow(self.p['severityClassBias'], j)
                classProbF = baseProbF*math.pow(self.p['severityClassBias'], j)
                classProbR = baseProbR*math.pow(self.p['severityClassBias'], j)
                g = sharesMalesByAgeAndIncome[i][j] + (1.0-sharesMalesByAgeAndIncome[i][j])*self.p['severityGenderBias']
                maleProbS = classProbS/g
                maleProbH = classProbH/g
                maleProbI = classProbI/g
                maleProbF = classProbF/g
                maleProbR = classProbR/g
                probClassAgeS.append([maleProbS, maleProbS*self.p['severityGenderBias']])
                probClassAgeH.append([maleProbH, maleProbH*self.p['severityGenderBias']])
                probClassAgeI.append([maleProbI, maleProbI*self.p['severityGenderBias']])
                probClassAgeF.append([maleProbF, maleProbF*self.p['severityGenderBias']])
                probClassAgeR.append([maleProbR, maleProbR*self.p['severityGenderBias']])
                
                if i == 8:
                    print 'Hospitalization probs for class ' + str(j) + ': ' + str([maleProbH, maleProbH*self.p['severityGenderBias']])
                    # pdb.set_trace()
            
                    
            self.probSymptomatic.append(probClassAgeS)
            self.probsHospitalization.append(probClassAgeH)
            self.probsIntensiveCare.append(probClassAgeI)
            self.intubatedFatalityRatio.append(probClassAgeF)
            self.infectionFatalityRatio.append(probClassAgeR)
        
        # pdb.set_trace()
        recoveryDays = [int(math.ceil(np.random.lognormal(self.p['meanRecovery'], self.p['sdRecovery']))) for i in range(10000)]
        recoveryDays.sort()
        self.recoveryPeriods = []
        start = 0
        end = 0
        for i in self.p['symptomsProbabilities']:
            end += int(i*10000)
            self.recoveryPeriods.append(recoveryDays[start:end])
            start = end
        
        # Compute age-classes shares
        ageShares = []
        numClasses = int(self.contacts.shape[1])
        if (numClasses%2) != 0:
            numClasses -= 1
        for i in range(0, numClasses, 2):
            ageClass_1 = len([x for x in self.pop.livingPeople if x.interactionAgeClass == i])
            ageClass_2 = len([x for x in self.pop.livingPeople if x.interactionAgeClass == i+1])
            share_1 = float(ageClass_1)/float(ageClass_1+ageClass_2)
            share_2 = float(ageClass_2)/float(ageClass_1+ageClass_2)
            shares = [share_1, share_2]
            ageShares.extend(shares)
            
        # 9-classes contacts matrix
        newMatrix = []
        for i in range(int(self.contacts.shape[1])):
            newColumn = []
            col = list(self.contacts[i])
            numCol = int(len(col))
            if (numCol%2) != 0:
                numCol -= 1
            for j in xrange(0, numCol, 2):
                newColumn.append(col[j]+col[j+1])
            if (len(col)%2) != 0:
                newColumn.append(col[-1])
            newMatrix.append(newColumn)
        self.contactsMatrix = []
        numCol = int(len(newMatrix))
        if (numCol%2) != 0:
            numCol -= 1
        for i in range(0, numCol, 2):
            self.contactsMatrix.append([(a*ageShares[i] + b*ageShares[i+1]) for a, b in zip(newMatrix[i], newMatrix[i+1])])
        if (len(newMatrix)%2) != 0:
            self.contactsMatrix.append(newMatrix[-1])
        

    def checkHouseholds(self, n):
        
#        for member in self.pop.livingPeople:
#            if member.partner != None and member.house != member.partner.house:
#                print 'Step: ' + str(n)
#                print 'Couple not living together'
#                print member.id
#                print member.dead
#                print member.independentStatus
#                print member.yearMarried
#                print member.partner.id
#                print member.partner.partner.id
#                print member.partner.dead
#                print member.partner.independentStatus
#                print member.partner.yearMarried
#                sys.exit()
    
        for house in self.map.occupiedHouses:
            
            household = house.occupants
            
#            if len(household) != len(set(household)):
#                print 'Step: ' + str(n)
#                print 'Error: person counted twice'
#                sys.exit()
                
#            if len(household) == 0:
#                print 'Step: ' + str(n)
#                print 'Error: occupied house is empty!'
#                sys.exit()
                
            married = [x for x in household if x.partner != None]
            
#            if len(married) > 2:
#                print 'Step: ' + str(n)
#                print 'Error: more than a couple in a house'
#                for member in married:
#                    print member.id
#                    print member.age
#                    print member.status
#                    print member.independentStatus
#                    print member.classRank
#                    print member.sex
#                    print member.income
#                    print member.careNeedLevel
#                    print 'Person partner id: ' + str(member.partner.id)
#                sys.exit()
                
#            if len(married) == 1:
#                print 'Step: ' + str(n)
#                print 'Error: married person not living with partner'
#                sys.exit()
           
            independentPeople = [x for x in household if x.independentStatus == True]
            
#            if len(independentPeople) == 0:
#                print 'Error: no independent people in the house'
#                print 'Step: ' + str(n)
#                for member in household:
#                    print member.id
#                    print member.age
#                    print member.status
#                    print member.classRank
#                    print member.sex
#                    print member.income
#                    print member.careNeedLevel
#                    print 'Father: ' + str(member.father.id)
#                    print member.father.dead
#                    print member.father.deadYear
#                    print member.father.yearMarried
#                    print member.father.yearDivorced
#                    print 'Mother: ' + str(member.mother.id)
#                    print member.mother.dead
#                    print member.mother.deadYear
#                    print member.mother.yearMarried
#                    print member.mother.yearDivorced
#                    
#                    if member.partner != None:
#                        print 'Person partner id: ' + str(member.partner.id)
#                    if member.mother.partner != None:
#                        print 'Person mother partner id: ' + str(member.mother.partner.id)
#                        print 'Person mother partner children: ' + str([x.id for x in member.mother.partner.children])
#                        if member.mother.partner.partner != None:
#                            print 'Person father partner id: ' + str(member.mother.partner.partner.id)
#                    if member.father.partner != None:
#                        print 'Person father partner id: ' + str(member.father.partner.id)
#                        print 'Person father partner children: ' + str([x.id for x in member.father.partner.children])
#                        if member.father.partner.partner != None:
#                            print 'Person father partner partner id: ' + str(member.father.partner.partner.id)
#                sys.exit()
                
            
####################   doDeath - SES version    ################################################
    def computeClassShares(self):
        
        self.socialClassShares[:] = []
        self.careNeedShares[:] = []
        peopleWithRank = [x for x in self.pop.livingPeople if x.classRank != -1]
        numPop = float(len(peopleWithRank))
        for c in range(int(self.p['numberClasses'])):
            classPop = [x for x in peopleWithRank if x.classRank == c]
            numclassPop = float(len(classPop))
            self.socialClassShares.append(numclassPop/numPop)
            
            needShares = []
            for b in range(int(self.p['numCareLevels'])):
                needPop = [x for x in classPop if x.careNeedLevel == b]
                numNeedPop = float(len(needPop))
                if numclassPop > 0:
                    needShares.append(numNeedPop/numclassPop)
                else:
                    needShares.append(0.0)
            self.careNeedShares.append(needShares)    
            
        print self.socialClassShares
    
    def deathProb(self, baseRate, person):  ##, shareUnmetNeed, classPop):
        
        classRank = person.classRank
        if person.status == 'child' or person.status == 'student':
            classRank = person.parentsClassRank
        
        if person.sex == 'male':
            mortalityBias = self.p['maleMortalityBias']
        else:
            mortalityBias = self.p['femaleMortalityBias']
        
        deathProb = baseRate
        
        a = 0
        for i in range(int(self.p['numberClasses'])):
            a += self.socialClassShares[i]*math.pow(mortalityBias, i)
            
        if a > 0:
            
            lowClassRate = baseRate/a
            
            classRate = lowClassRate*math.pow(mortalityBias, classRank)
            
            deathProb = classRate
           
            b = 0
            for i in range(int(self.p['numCareLevels'])):
                b += self.careNeedShares[classRank][i]*math.pow(self.p['careNeedBias'], (self.p['numCareLevels']-1) - i)
                
            if b > 0:
                
                higherNeedRate = classRate/b
               
                deathProb = higherNeedRate*math.pow(self.p['careNeedBias'], (self.p['numCareLevels']-1) - person.careNeedLevel) # deathProb
      
        # Add the effect of unmet care need on mortality rate for each care need level
        
        ##### Temporarily by-passing the effect of Unmet Care Need   #############
        
#        a = 0
#        for x in classPop:
#            a += math.pow(self.p['unmetCareNeedBias'], 1-x.averageShareUnmetNeed)
#        higherUnmetNeed = (classRate*len(classPop))/a
#        deathProb = higherUnmetNeed*math.pow(self.p['unmetCareNeedBias'], 1-shareUnmetNeed)
        
        return deathProb
    
    def deathProb_UCN(self, baseRate, sex, classRank, needLevel, shareUnmetNeed, classPop):
        
        if sex == 'male':
            mortalityBias = self.p['maleMortalityBias']
        else:
            mortalityBias = self.p['femaleMortalityBias']
        
        a = 0
        for i in range(int(self.p['numberClasses'])):
            a += self.socialClassShares[i]*math.pow(mortalityBias, i)
        lowClassRate = baseRate/a
        
        classRate = lowClassRate*math.pow(mortalityBias, classRank)
       
        a = 0
        for i in range(int(self.p['numCareLevels'])):
            a += self.careNeedShares[classRank][i]*math.pow(self.p['careNeedBias'], (self.p['numCareLevels']-1) - i)
        higherNeedRate = classRate/a
       
        classRate = higherNeedRate*math.pow(self.p['careNeedBias'], (self.p['numCareLevels']-1) - needLevel) # deathProb
      
        # Add the effect of unmet care need on mortality rate for each care need level
        
        ##### Temporarily by-passing the effect of Unmet Care Need   #############
        
        a = 0
        for x in classPop:
            a += math.pow(self.p['unmetCareNeedBias'], 1-x.averageShareUnmetNeed)
        higherUnmetNeed = (classRate*len(classPop))/a
        deathProb = higherUnmetNeed*math.pow(self.p['unmetCareNeedBias'], 1-shareUnmetNeed)
        
        return deathProb
    
    def doDeaths(self):
        
        preDeath = len(self.pop.livingPeople)
        
        deaths = [0, 0, 0, 0, 0]
        """Consider the possibility of death for each person in the sim."""
        for person in self.pop.livingPeople:
            age = person.age
            
            ####     Death process with histroical data  after 1950   ##################
            if self.year >= 1950:
                if age > 109:
                    age = 109
                if person.sex == 'male':
                    rawRate = self.death_male[age, self.year-1950]
                if person.sex == 'female':
                    rawRate = self.death_female[age, self.year-1950]
                    
                classPop = [x for x in self.pop.livingPeople if x.careNeedLevel == person.careNeedLevel]
                
                dieProb = self.deathProb(rawRate, person)
                
                person.lifeExpectancy = max(90-person.age, 5.0)
                # dieProb = self.deathProb_UCN(rawRate, person, person.averageShareUnmetNeed, classPop)

            #############################################################################
            
                if random.random() < dieProb:
                    person.dead = True
                    person.deadYear = self.year
                    person.house.occupants.remove(person)
                    if len(person.house.occupants) == 0:
                        self.map.occupiedHouses.remove(person.house)
                    if person.partner != None:
                        person.partner.partner = None
    
            else: 
                #######   Death process with made-up rates  ######################
                babyDieProb = 0.0
                if age < 1:
                    babyDieProb = self.p['babyDieProb']
                if person.sex == 'male':
                    ageDieProb = (math.exp(age/self.p['maleAgeScaling']))*self.p['maleAgeDieProb'] 
                else:
                    ageDieProb = (math.exp(age/self.p['femaleAgeScaling']))* self.p['femaleAgeDieProb']
                rawRate = self.p['baseDieProb'] + babyDieProb + ageDieProb
                
                classPop = [x for x in self.pop.livingPeople if x.careNeedLevel == person.careNeedLevel]
                
                dieProb = self.deathProb(rawRate, person)
                
                person.lifeExpectancy = max(90-person.age, 5.0)
                #### Temporarily by-passing the effect of unmet care need   ######
                # dieProb = self.deathProb_UCN(rawRate, person.parentsClassRank, person.careNeedLevel, person.averageShareUnmetNeed, classPop)
                
                if random.random() < dieProb:
                    person.dead = True
                    person.deadYear = self.year
                    deaths[person.classRank] += 1
                    person.house.occupants.remove(person)
                    if len(person.house.occupants) == 0:
                        self.map.occupiedHouses.remove(person.house)
                    if person.partner != None:
                        person.partner.partner = None
                  
        self.pop.livingPeople[:] = [x for x in self.pop.livingPeople if x.dead == False]
        
        postDeath = len(self.pop.livingPeople)
        
        print('the number of deaths is: ' + str(preDeath - postDeath))            

    def doCareTransitions(self):
        """Consider the possibility of each person coming to require care."""
        peopleNotInCriticalCare = [x for x in self.pop.livingPeople if x.careNeedLevel < self.p['numCareLevels']-1]
        for person in peopleNotInCriticalCare:
            age = self.year - person.birthdate
            if person.sex == 'male':
                ageCareProb = ( ( math.exp( age /
                                            self.p['maleAgeCareScaling'] ) )
                               * self.p['personCareProb'] )
            else:
                ageCareProb = ( ( math.exp( age /
                                           self.p['femaleAgeCareScaling'] ) )
                               * self.p['personCareProb'] )
            careProb = self.p['baseCareProb'] + ageCareProb
            
            if random.random() < careProb:
                multiStepTransition = random.random()
                if multiStepTransition < self.p['cdfCareTransition'][0]:
                    person.careNeedLevel += 1
                elif multiStepTransition < self.p['cdfCareTransition'][1]:
                    person.careNeedLevel += 2
                elif multiStepTransition < self.p['cdfCareTransition'][2]:
                    person.careNeedLevel += 3
                else:
                    person.careNeedLevel += 4
                    
                if person.careNeedLevel >= self.p['numCareLevels']:
                    person.careNeedLevel = int(self.p['numCareLevels'] - 1)
                            

    def doCareTransitions_UCN(self):
        """Consider the possibility of each person coming to require care."""
        peopleNotInCriticalCare = [x for x in self.pop.livingPeople if x.careNeedLevel < self.p['numCareLevels']-1]
        for person in peopleNotInCriticalCare:
            age = self.year - person.birthdate
            if person.sex == 'male':
                ageCareProb = ( ( math.exp( age /
                                            self.p['maleAgeCareScaling'] ) )
                               * self.p['personCareProb'] )
            else:
                ageCareProb = ( ( math.exp( age /
                                           self.p['femaleAgeCareScaling'] ) )
                               * self.p['personCareProb'] )
            baseProb = self.p['baseCareProb'] + ageCareProb
            
            baseProb = self.baseRate(self.p['careBias'], baseProb)
            
            unmetNeedFactor = 1.0/math.exp(self.p['unmetNeedExponent']*person.averageShareUnmetNeed)
            
            
            classRank = person.classRank
            if person.status == 'child' or person.status == 'student':
                classRank = person.parentsClassRank
            
            careProb = baseProb*math.pow(self.p['careBias'], classRank)/unmetNeedFactor 
            
            if np.random.random() < careProb:
                baseTransition = self.baseRate(self.p['careBias'], 1.0-self.p['careTransitionRate'])
                if baseTransition >= 1.0:
                    print 'Error: base transition >= 1'
                    # sys.exit()
                    
                    
                if person.careNeedLevel > 0:
                    unmetNeedFactor = 1.0/math.exp(self.p['unmetNeedExponent']*person.averageShareUnmetNeed)
                else:
                    unmetNeedFactor = 1.0
                transitionRate = (1.0 - baseTransition*math.pow(self.p['careBias'], classRank))*unmetNeedFactor
                
                stepCare = 1
                bound = transitionRate
                
                while np.random.random() > bound and stepCare < self.p['numCareLevels'] - 1:
                    stepCare += 1
                    bound += (1.0-bound)*transitionRate
                person.careNeedLevel += stepCare
                    
                if person.careNeedLevel >= self.p['numCareLevels']:
                    person.careNeedLevel = int(self.p['numCareLevels'] - 1)
                if person.careNeedLevel > 1:
                    person.wage = 0
    
    def baseRate(self, bias, cp):
        a = 0
        for i in range(int(self.p['numberClasses'])):
            a += self.socialClassShares[i]*math.pow(bias, i)
        baseRate = cp/a
        return (baseRate)
    
    def updateUnmetCareNeed(self):
        
        for house in self.map.occupiedHouses:
            house.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in house.occupants])
            house.careNetwork.clear()
            house.suppliers[:] = []
            house.demandNetwork.clear()
            house.receivers[:] = []
        
        for person in self.pop.livingPeople:
            person.careNetwork.clear()
            person.suppliers[:] = []
            person.inHouseInformalCare = 0
            person.externalInformalCare = 0
            if person.hoursSocialCareDemand > 0:
                person.cumulativeUnmetNeed *= self.p['unmetCareNeedDiscountParam']
                person.cumulativeUnmetNeed += person.unmetSocialCareNeed
                person.totalDiscountedShareUnmetNeed *= self.p['shareUnmetNeedDiscountParam']
                person.totalDiscountedTime *= self.p['shareUnmetNeedDiscountParam']
                person.totalDiscountedShareUnmetNeed += person.unmetSocialCareNeed/person.hoursSocialCareDemand
                person.totalDiscountedTime += 1
                person.averageShareUnmetNeed = person.totalDiscountedShareUnmetNeed/person.totalDiscountedTime
        
    def allocateChildCare(self):
        
        self.costTaxFreeChildCare = 0

        receivers = [x for x in self.map.occupiedHouses if x.totalChildCareNeed > 0]
        for receiver in receivers:
            self.computeChildCareNetworkSupply(receiver)
        residualReceivers = [x for x in receivers if x.networkSupply > 0]
        
        while len(residualReceivers) > 0:
    
            #################    Check if transfer is done: Pre need and supply   ######################################
            
            preChildCareNeed = sum([x.totalChildCareNeed for x in residualReceivers])
            preNetworkSupply = sum([x.networkSupply for x in residualReceivers])
            
            #########################################################################################
            
            # An household is selected depending on total child care need
            childCareNeeds = [x.totalChildCareNeed for x in residualReceivers]
            probReceivers = [i/sum(childCareNeeds) for i in childCareNeeds]
            receiver = np.random.choice(residualReceivers, p = probReceivers)
            
#            print 'High price care: ' + str(receiver.highPriceChildCare)
#            print 'Low price care: ' + str(receiver.lowPriceChildCare)
#            
#            if receiver.highPriceChildCare + receiver.lowPriceChildCare == 0:
#                print 'Error: receiver has not child care!'
#                sys.exit()
#            
#            print ''
            
            ###    Individual check   ##########################
            
            preReceiverCareNeed = receiver.totalChildCareNeed
            
            ###########################################################
            case = 0
            
            income = self.computeHouseholdIncome(receiver)
            # print 'Household income: ' + str(income)
            residualIncome = receiver.residualIncomeForChildCare
            
#            if int(receiver.residualIncomeForChildCare) > int(self.computeHouseholdIncome(receiver)):
#                print 'Error: residual income greater than income (1)'
#                print receiver.residualIncomeForChildCare
#                print self.computeHouseholdIncome(receiver)
#                print receiver.householdFormalSupplyCost
#                print receiver.initialResidualIncomeForChildCare
#                
#                for x in receiver.occupants:
#                    print x.status
#                    print x.careNeedLevel
#                    print x.potentialIncome
#                    print x.residualWorkingHours
#                    print x.income
#                
#                sys.exit()
                
            # print 'Residual Income for child care: ' + str(residualIncome) 
            childcareExpense = receiver.householdFormalSupplyCost
            # print 'Childcare expense: ' + str(childcareExpense)
            total = residualIncome+childcareExpense
            # print 'Total: ' + str(total)
            
#            if int(income) != int(total):
#                print 'Error receiver'
#                print receiver.id
#                print receiver.totalChildCareNeed
#                print receiver.highPriceChildCare
#                print receiver.lowPriceChildCare
#                print sum(receiver.networkInformalSupplies)
#                if receiver.totalSocialCareNeed > 0:
#                    print sum(receiver.networkInformalSupplies)/receiver.totalSocialCareNeed
#                print 'Household income: ' + str(income)
#                print 'Residual Income for child care: ' + str(residualIncome) 
#                print 'Childcare expense: ' + str(childcareExpense)
#                print 'Total: ' + str(total)
#                print receiver.networkSupply
#                for x in receiver.occupants:
#                    print x.status
#                    print x.careNeedLevel
#                    if x.status == 'worker':
#                        print x.potentialIncome
#                        print x.residualWorkingHours
#                    print x.income
#                sys.exit()
            
            
            if receiver.highPriceChildCare > 0:
            # If the receiver has childcare need more expensive than social care, it will try to satisfy it with informal care. 
            # Only if there in no enough informal care the formal care will be suppllied.
            # Therefore, the supplier will be chosen according to the informal care availability, and only if there is no informal care availability left,
            # it will be chosen according to the formal care availability.
                # Select a supplier based on availability of informal supply.
                if sum(receiver.networkInformalSupplies) > 0:
                    
#                    print 'Receiver check'
#                    print receiver.totalChildCareNeed
#                    print sum(receiver.networkInformalSupplies)
#                    print receiver.networkInformalSupplies
                    
                    sumSupply = float(sum(receiver.networkInformalSupplies))
                    # A supplier is selected based on informal care availability
                    probSuppliers = [float(i)/sumSupply for i in receiver.networkInformalSupplies]
#                    print probSuppliers
#                    print ''
                    supplier = np.random.choice(receiver.suppliers, p = probSuppliers)
                    
                    case = 1
#                    print 'tranfer child care: formal (1)'
#                    print sum(receiver.networkInformalSupplies)
                    
                    self.transferInformalChildCare(receiver, supplier)
                    
                    
                elif receiver.formalChildCareSupply > 0:
                    supplier = receiver
                    # Formal supply: the supplier can only be the household itself.
                    case = 2
#                    print 'tranfer child care: formal (2)'
#                    print receiver.formalChildCareSupply
                    
                    self.outOfIncomeChildCare(receiver)
                    
                    
                    
            elif receiver.lowPriceChildCare > 0:
                # In this case, a random selection based on relative availability will be done until the ratio between 
                # the available informal care and the household's social care is above 1, i.e. there is enough informal care.
                # As the ratio goes below 1, only out-of-income care will be supplied.
                totalCare = sum(receiver.networkInformalSupplies) + receiver.formalChildCareSupply
                typesOfCare = ['informal care', 'formal care']
                if totalCare > 0:
                    probInformalCare = max(sum(receiver.networkInformalSupplies)-receiver.totalUnmetSocialCareNeed, 0)/totalCare
                    probs = [probInformalCare, 1-probInformalCare]
                    # print 'Care probs: ' + str(probs)
                    # print 'Formal supply: ' + str(receiver.formalChildCareSupply)
                    care = np.random.choice(typesOfCare, p = probs)
                    if receiver.formalChildCareSupply == 0 or care == 'informal care':
                        
#                        print 'Receiver check'
#                        print receiver.totalChildCareNeed
#                        print sum(receiver.networkInformalSupplies)
#                        print receiver.networkInformalSupplies
#                        print care
                        sumSupply = float(sum(receiver.networkInformalSupplies))
                    # A supplier is selected based on informal care availability
                        probSuppliers = [float(i)/sumSupply for i in receiver.networkInformalSupplies]
                        supplier = np.random.choice(receiver.suppliers, p = probSuppliers)
                    
                        case = self.transferInformalChildCare(receiver, supplier)
                    else:
                        supplier = receiver
                        case = 5
                        self.outOfIncomeChildCare(receiver)
#            else:
#                
#                print 'Error: receiver does not have child care need'
#                sys.exit()
                    
             ###    Individual check   ##########################
             
#            if case == 0:
#                print 'Error: no allocation function called!'
#                sys.exit()
            
            postReceiverCareNeed = receiver.totalChildCareNeed
            
            if postReceiverCareNeed >= preReceiverCareNeed:
                print case
                print 'Error: child care iteration withount allocation!'
                # sys.exit()
            
            ########################################################### 
        
            receivers = [x for x in self.map.occupiedHouses if x.totalChildCareNeed > 0]
            
            for otherReceiver in receivers:
                if otherReceiver == receiver:
                    continue
                self.updateChildCareNetworkSupply(otherReceiver, supplier, 4)
           
            residualReceivers = [x for x in receivers if x.networkSupply > 0]
            
            #################    Check if transfer is done: Post need and supply   ######################################
            
            postChildCareNeed = sum([x.totalChildCareNeed for x in residualReceivers])
            postNetworkSupply = sum([x.networkSupply for x in residualReceivers])
            
            # print [x.totalChildCareNeed for x in residualReceivers]
            
            
#            print postChildCareNeed
#            print preChildCareNeed
#            print ''
            if postChildCareNeed >= preChildCareNeed:
                print 'Error: child care iteration withount allocation!'
                # sys.exit()
                
            #########################################################################################
            
#        if self.year == 1862:
#            sys.exit()
        
    def allocateSocialCare(self):
        
        self.inHouseInformalCare = 0
        
        self.computeResidualIncomeForSocialCare() # self.computeResidualIncomeForSocialCare()
        
        # self.householdCareNetwork_netSupply()
        
        receivers = [x for x in self.map.occupiedHouses if x.totalUnmetSocialCareNeed > 0]
        for receiver in receivers:
            self.computeSocialCareNetworkSupply_W(receiver) ###  Add social care from wealth.....
            
            #######    Temporarily excluding out-of-wealth care   ##########
            # self.computeSocialCareNetworkSupply_W(receiver)
            
        residualReceivers = [x for x in receivers if x.networkSupply > 0]
        
        
        
        while len(residualReceivers) > 0:
            
            #################    Check if transfer is done: Pre need and supply   ######################################
            
            preSocialCareNeed = sum([x.totalUnmetSocialCareNeed for x in residualReceivers])
            preNetworkSupply = sum([x.networkSupply for x in residualReceivers])
            
            #########################################################################################
            
            
            socialCareNeeds = [x.totalUnmetSocialCareNeed for x in residualReceivers]
            probReceivers = [i/sum(socialCareNeeds) for i in socialCareNeeds]
            receiver = np.random.choice(residualReceivers, p = probReceivers)
            
            if self.socialCareNetwork.has_node(receiver) == False:
                self.socialCareNetwork.add_node(receiver, internalSupply = 0)
            
            ###    Individual check   ##########################
            
            preReceiverCareNeed = receiver.totalUnmetSocialCareNeed
            
            ###########################################################
            suppliersWeights = []
            for supplier in receiver.suppliers:
                indexSupplier = receiver.suppliers.index(supplier)
                informalSupply = receiver.networkInformalSupplies[indexSupplier]
                formalSocialCare = receiver.networkFormalSocialCareSupplies[indexSupplier]
                careFromWealth = 0
                if receiver == supplier:
                    careFromWealth = receiver.careSupplyFromWealth
                totalFormaCare = formalSocialCare + careFromWealth
                informalFactor = math.pow(informalSupply, self.p['betaInformalCare'])
                formalFactor = math.pow(totalFormaCare, (1-self.p['betaInformalCare']))
                suppliersWeights.append(informalFactor+formalFactor)
            
            probSuppliers = [i/sum(suppliersWeights) for i in suppliersWeights]
            
            
            supplier = np.random.choice(receiver.suppliers, p = probSuppliers)
            
            if self.socialCareNetwork.has_node(supplier) == False:
                self.socialCareNetwork.add_node(supplier, internalSupply = 0)
            
            if self.socialCareNetwork.has_edge(receiver, supplier) == False:
                self.socialCareNetwork.add_edge(receiver, supplier, careTransferred = 0)
        
            self.transferSocialCare_W(receiver, supplier) # self.transferSocialCare(receiver, supplier)
            
            ###    Individual check   ##########################
            
            postReceiverCareNeed = receiver.totalUnmetSocialCareNeed
            
            if postReceiverCareNeed >= preReceiverCareNeed:
                print 'Error: social care iteration withount allocation!'
                sys.exit()
            
            ########################################################### 
            
            receivers = [x for x in self.map.occupiedHouses if x.totalUnmetSocialCareNeed > 0]
            for otherReceiver in receivers:
                if otherReceiver == receiver:
                    continue
                self.updateSocialCareNetworkSupply_W(otherReceiver, supplier, 0) # self.updateSocialCareNetworkSupply(receiver, supplier, 0)
            residualReceivers = [x for x in receivers if x.networkSupply > 0]
            
            
            #################    Check if transfer is done: Post need and supply   ######################################
            
            postSocialCareNeed = sum([x.totalUnmetSocialCareNeed for x in residualReceivers])
            postNetworkSupply = sum([x.networkSupply for x in residualReceivers])
        
            if postSocialCareNeed >= preSocialCareNeed:
                print 'Error: social care iteration withount allocation!'
                # sys.exit()
                
            #########################################################################################
    
    
    def allocateSocialCare_Ind(self):
        
        self.inHouseInformalCare = 0
        
        self.computeResidualIncomeForSocialCare()
        
        receivers = [x for x in self.pop.livingPeople if x.unmetSocialCareNeed > 0]
        self.totInHouseSupply = 0
        self.totExternalSupply = 0
        for receiver in receivers:
            self.computeSocialCareNetworkSupply_Ind(receiver)

        self.totInHouseSupply = sum([x.inHouseInformalCare for x in receivers])
        self.totExternalSupply = sum([x.externalInformalCare for x in receivers])
        
        self.inHouseCareSupplyRatio = 0
        if self.totInHouseSupply+self.totExternalSupply > 0:
            self.inHouseCareSupplyRatio = float(self.totInHouseSupply)/float(self.totInHouseSupply+self.totExternalSupply)
        
        print 'Share of in-house care supply: ' + str(self.inHouseCareSupplyRatio)
        
        residualReceivers = [x for x in receivers if x.networkSupply > 0]
        
        while len(residualReceivers) > 0:
            
            #################    Check if transfer is done: Pre need and supply   ######################################
            
            preSocialCareNeed = sum([x.unmetSocialCareNeed for x in residualReceivers])
            preNetworkSupply = sum([x.networkSupply for x in residualReceivers])
            
            #########################################################################################
            
            
            socialCareNeeds = [x.unmetSocialCareNeed for x in residualReceivers]
            probReceivers = [i/sum(socialCareNeeds) for i in socialCareNeeds]
            
            if sum(probReceivers) == 0.0:
                print 'Prob receivers: ' + str(sum(probReceivers))
                print 'Unmet social care: ' + str([x.unmetSocialCareNeed for x in residualReceivers])
                print 'Network supplies: ' + str([x.networkSupply for x in residualReceivers])
                sys.exit()

                
            receiver = np.random.choice(residualReceivers, p = probReceivers)
            
            if self.socialCareNetwork.has_node(receiver) == False:
                self.socialCareNetwork.add_node(receiver, internalSupply = 0)
            
            ###    Individual check   ##########################
            
            preReceiverCareNeed = receiver.unmetSocialCareNeed
            
            ###########################################################
            
            suppliersWeights = list(receiver.weightedTotalSupplies)
            potentialSuppliers = list(receiver.suppliers)
            suppliersWeights.append(receiver.careSupplyFromWealth)
            potentialSuppliers.append(receiver)
            
            probSuppliers = [i/sum(suppliersWeights) for i in suppliersWeights]
            supplier = np.random.choice(potentialSuppliers, p = probSuppliers)
            
            if self.socialCareNetwork.has_node(supplier) == False:
                self.socialCareNetwork.add_node(supplier, internalSupply = 0)
            
            if self.socialCareNetwork.has_edge(receiver, supplier) == False:
                self.socialCareNetwork.add_edge(receiver, supplier, careTransferred = 0)
        
            self.transferSocialCare_Ind(receiver, supplier) # self.transferSocialCare(receiver, supplier)
            
            ###    Individual check   ##########################
            
            postReceiverCareNeed = receiver.unmetSocialCareNeed
            
            if postReceiverCareNeed >= preReceiverCareNeed:
                print 'Error: social care iteration withount allocation!'
                sys.exit()
            
            ########################################################### 
            
            receivers = [x for x in self.pop.livingPeople if x.unmetSocialCareNeed > 0]
            for otherReceiver in receivers:
                if otherReceiver == receiver:
                    continue
                self.updateSocialCareNetworkSupply_Ind(otherReceiver, supplier, 0) # self.updateSocialCareNetworkSupply(receiver, supplier, 0)
            residualReceivers = [x for x in receivers if x.networkSupply > 0]
            
            
            #################    Check if transfer is done: Post need and supply   ######################################
            
            postSocialCareNeed = sum([x.unmetSocialCareNeed for x in residualReceivers])
            postNetworkSupply = sum([x.networkSupply for x in residualReceivers])
        
            if postSocialCareNeed >= preSocialCareNeed:
                print 'Error: social care iteration withount allocation!'
                # sys.exit()
                
            #########################################################################################
    
    
    def transferSocialCare(self, receiver, supplier):
        
        distance = 0
        if receiver != supplier:
            distance = receiver.careNetwork[receiver][supplier]['distance']
        
        indexSupplier = receiver.suppliers.index(supplier)
        informalSupply = receiver.networkInformalSupplies[indexSupplier]
        formalSocialCare = receiver.networkFormalSocialCareSupplies[indexSupplier]
        
        # Select kind of care based on supplier availability
        kindsOfCare = ['informal care', 'out-of-income care'] # Add an 'out-of-wealth' care............
        care = 'out-of-income care'
        if supplier.town == receiver.town:
            careWeights = [informalSupply, formalSocialCare]
            careProbs = [x/sum(careWeights) for x in careWeights]
            care = np.random.choice(kindsOfCare, p = careProbs) 
        
        if care == 'informal care':
        
            household = list(supplier.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            # If informal care is provided, it satisfies the most expensive cumulated child care need.
            for member in householdCarers:
                member.residualInformalSupply = member.residualInformalSupplies[distance]
            householdCarers.sort(key=operator.attrgetter("residualInformalSupply"), reverse=True)
            
            residualCare = min(self.p['quantumCare'], receiver.totalUnmetSocialCareNeed)
            if residualCare < 1.0:
                residualCare = 1.0
            
            careTransferred = 0
            for i in householdCarers:
                careForNeed = min(i.residualInformalSupply, residualCare)
                if careForNeed < 1.0:
                    careForNeed = 1.0
                if i in receiver.occupants and careForNeed > 0:
                    i.careForFamily = True
                i.socialWork += careForNeed
                if careForNeed > 0:
                    for j in range(4):
                        i.residualInformalSupplies[j] -= careForNeed
                        i.residualInformalSupplies[j] = float(max(int(i.residualInformalSupplies[j]), 0))
                    careTransferred += careForNeed
                    residualCare -= careForNeed
                    if residualCare <= 0:
                        break
                    
            if receiver == supplier:
                self.inHouseInformalCare += careTransferred
                
            if  receiver != supplier:
                receiver.networkSupport += careTransferred
            
            receiverOccupants = list(receiver.occupants)
            careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
            careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
            residualCare = careTransferred
            for person in careTakers:
                personalCare = min(person.unmetSocialCareNeed, residualCare)
                person.unmetSocialCareNeed -= personalCare
                person.unmetSocialCareNeed = float(max(int(person.unmetSocialCareNeed), 0))
                person.informalSocialCareReceived += personalCare
                residualCare -= personalCare
                if residualCare <= 0:
                    break
            receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
            receiver.informalSocialCareReceived += careTransferred
            
            self.updateSocialCareNetworkSupply(receiver, supplier, 1)
            
        elif care == 'out-of-income care':
            household = list(supplier.occupants)
            employed = [x for x in household if x.status == 'worker' and x.availableWorkingHours > 0 and x.maternityStatus == False]
            
            maxFormalCare = receiver.networkFormalSocialCareSupplies[indexSupplier]
            
            residualCare = min(self.p['quantumCare'], receiver.totalUnmetSocialCareNeed, maxFormalCare)
            if residualCare < 1.0:
                residualCare = 1.0
            
            costQuantumSocialCare = self.socialCareCost(supplier, residualCare)
            priceSocialCare = costQuantumSocialCare/residualCare
            
            if supplier.town != receiver.town or len(employed) == 0: # Only formal care is possible
                
#                print 'Supplier id: ' + str(supplier.id)
#                print 'Pre income for social care: ' + str(supplier.residualIncomeForSocialCare)
#                print 'Cost quantum social care: ' + str(costQuantumSocialCare)
                
                for j in range(4):
                    supplier.residualIncomeForSocialCare[j] -= costQuantumSocialCare
                    supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0.0)
                supplier.householdFormalSupplyCost += costQuantumSocialCare
                
                careTransferred = 0
                if len(employed) > 0:
                    employed.sort(key=operator.attrgetter("wage"), reverse=True)
                    residualCare = self.p['quantumCare']
                    for worker in employed:
                        maxCare = costQuantumSocialCare/worker.wage
                        workerCare = min(maxCare, residualCare)
                        if workerCare < 1.0:
                            workerCare = 1.0
                        worker.availableWorkingHours -= workerCare
                        worker.availableWorkingHours = float(max(int(worker.availableWorkingHours), 0))
                        careTransferred += workerCare
                        costQuantumSocialCare -= workerCare*worker.wage
                        residualCare -= workerCare
                        if residualCare <= 0:
                            break
                
                # print 'Post income for social care: ' + str(supplier.residualIncomeForSocialCare)
                
                receiverOccupants = list(receiver.occupants)                
                careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
                careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
                residualCare = careTransferred
                for person in careTakers:
                    personalCare = min(person.unmetSocialCareNeed, residualCare)
                    person.unmetSocialCareNeed -= personalCare
                    person.unmetSocialCareNeed = float(max(int(person.unmetSocialCareNeed), 0))
                    person.formalSocialCareReceived += personalCare
                    residualCare -= personalCare
                    if residualCare <= 0:
                        break
                receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
                receiver.formalSocialCareReceived += careTransferred
                
                self.updateSocialCareNetworkSupply(receiver, supplier, 2)
                
            else:
                # Select the worker with the lowest pay
                employed.sort(key=operator.attrgetter("wage"))
                
                carer = employed[0]
                if carer.wage > priceSocialCare: # In this case, it is more convenient to pay for formal care
                    for j in range(4):
                        supplier.residualIncomeForSocialCare[j] -= costQuantumSocialCare
                        supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0)
                    supplier.householdFormalSupplyCost += costQuantumSocialCare
                    
                    employed.sort(key=operator.attrgetter("wage"), reverse=True)
                    careToAllocate = residualCare
                    careTransferred = 0
                    for worker in employed:
                        maxCare = costQuantumSocialCare/worker.wage
                        workerCare = min(maxCare, careToAllocate)
                        if workerCare < 1.0:
                            workerCare = 1.0
                        careTransferred += workerCare
                        worker.availableWorkingHours -= workerCare
                        worker.availableWorkingHours = float(max(int(worker.availableWorkingHours), 0))
                        costQuantumSocialCare -= workerCare*worker.wage
                        careToAllocate -= workerCare
                        if careToAllocate <= 0:
                            break
                        
                    # employed[0].availableWorkingHours -= costQuantumSocialCare/employed[0].wage
                    # employed[0].availableWorkingHours = max(employed[0].availableWorkingHours, 0)
                    
                    receiverOccupants = list(receiver.occupants)
                    careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
                    careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
                    residualCare = careTransferred
                    for person in careTakers:
                        personalCare = min(person.unmetSocialCareNeed, residualCare)
                        person.unmetSocialCareNeed -= personalCare
                        person.unmetSocialCareNeed = float(max(int(person.unmetSocialCareNeed), 0))
                        person.formalSocialCareReceived += personalCare
                        residualCare -= personalCare
                        if residualCare <= 0:
                            break
                    receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
                    receiver.formalSocialCareReceived += careTransferred
                    
                    self.updateSocialCareNetworkSupply(receiver, supplier, 3)
                    
                else: # In this case, it is more convenient to take time off work to provide informal care.
                    print 'Case 4'
                    print 'Carer pre available working hours: ' + str(carer.availableWorkingHours)
                    
                    if receiver == supplier:
                        self.inHouseInformalCare += self.p['quantumCare']
                    
                    careTransferred = min(self.p['quantumCare'], carer.availableWorkingHours)
                    if careTransferred == 0:
                        print 'No residual working hours in carer'
                        sys.exit()
                    if careTransferred < 1.0:
                        careTransferred = 1.0
                    if carer in receiver.occupants and careTransferred > 0:
                        carer.careForFamily = True
                    incomeForCare = careTransferred*carer.wage
                    carer.availableWorkingHours -= careTransferred #self.p['quantumCare']
                    carer.availableWorkingHours = float(max(0, int(carer.availableWorkingHours)))
                    carer.residualWorkingHours -= careTransferred
                    carer.residualWorkingHours = float(max(0, int(carer.residualWorkingHours)))
                    carer.outOfWorkSocialCare += careTransferred
                    carer.socialWork += careTransferred
                    
                    print 'Care transferred: ' + str(careTransferred)
                    print 'Carer wage: ' + str(carer.wage)
                    print 'Carer post available working hours: ' + str(carer.availableWorkingHours)
                    print 'Pre-residual Income for care (4): ' + str(supplier.residualIncomeForSocialCare)
                    
                    for j in range(4):
                        supplier.residualIncomeForSocialCare[j] -= incomeForCare
                        supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0)
                        
                    print 'Post-residual Income for care (4): ' + str(supplier.residualIncomeForSocialCare)
                    
                    receiverOccupants = list(receiver.occupants)
                    careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
                    careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
                    residualCare = careTransferred
                    for person in careTakers:
                        personalCare = min(person.unmetSocialCareNeed, residualCare)
                        person.unmetSocialCareNeed -= personalCare
                        person.unmetSocialCareNeed = float(max(int(person.unmetSocialCareNeed), 0))
                        person.informalSocialCareReceived += personalCare
                        residualCare -= personalCare
                        if residualCare <= 0:
                            break
                    receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
                    receiver.informalSocialCareReceived += careTransferred
                    
                    self.updateSocialCareNetworkSupply(receiver, supplier, 4)
            
            
    def transferSocialCare_W(self, receiver, supplier):
        
        distance = 0
        if receiver != supplier:
            distance = receiver.careNetwork[receiver][supplier]['distance']
        
        indexSupplier = receiver.suppliers.index(supplier)
        informalSupply = receiver.networkInformalSupplies[indexSupplier]
        formalSocialCare = receiver.networkFormalSocialCareSupplies[indexSupplier]
        careFromWealth = 0
        if receiver == supplier:
            careFromWealth = receiver.careSupplyFromWealth
        totalFormaCare = formalSocialCare + careFromWealth
        informalFactor = math.pow(informalSupply, self.p['betaInformalCare'])
        formalFactor = math.pow(totalFormaCare, (1-self.p['betaInformalCare']))
        probInformalCare = informalFactor/(informalFactor+formalFactor)
        # Select kind of care based on supplier availability
        kindsOfCare = ['informal care', 'formal care'] # Add an 'out-of-wealth' care............
        care = 'out-of-income care'
        if supplier.town == receiver.town:
            # careWeights = [informalSupply, formalSocialCare, careFromWealth]
            # careProbs = [x/sum(careWeights) for x in careWeights]
            careProbs = [probInformalCare, 1-probInformalCare]
            care = np.random.choice(kindsOfCare, p = careProbs) 
        
        if care == 'informal care':
            
            
            household = list(supplier.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            # If informal care is provided, it satisfies the most expensive cumulated child care need.
            for member in householdCarers:
                member.residualInformalSupply = member.residualInformalSupplies[distance]
            householdCarers.sort(key=operator.attrgetter("residualInformalSupply"), reverse=True)
            
            residualCare = min(self.p['quantumCare'], receiver.totalUnmetSocialCareNeed)
            if residualCare < 1.0:
                residualCare = 1.0
            
            careTransferred = 0
            for i in householdCarers:
                careForNeed = min(i.residualInformalSupply, residualCare)
                if careForNeed < 1.0:
                    careForNeed = 1.0
                if i in receiver.occupants and careForNeed > 0:
                    i.careForFamily = True
                i.socialWork += careForNeed
                if careForNeed > 0:
                    for j in range(4):
                        i.residualInformalSupplies[j] -= careForNeed
                        i.residualInformalSupplies[j] = float(max(int(i.residualInformalSupplies[j]), 0))
                    careTransferred += careForNeed
                    residualCare -= careForNeed
                    if residualCare <= 0:
                        break
                    
            if receiver == supplier:
                self.inHouseInformalCare += careTransferred
                
            if  receiver != supplier:
                receiver.networkSupport += careTransferred
                
            if receiver != supplier:
                self.socialCareNetwork[receiver][supplier]['careTransferred'] += careTransferred
            else:
                self.socialCareNetwork.node[receiver]['internalSupply'] += careTransferred
            
            receiverOccupants = list(receiver.occupants)
            careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
            careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
            residualCare = careTransferred
            for person in careTakers:
                personalCare = min(person.unmetSocialCareNeed, residualCare)
                person.unmetSocialCareNeed -= personalCare
                person.unmetSocialCareNeed = float(max(int(person.unmetSocialCareNeed), 0))
                person.informalSocialCareReceived += personalCare
                residualCare -= personalCare
                if residualCare <= 0:
                    break
            receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
            receiver.informalSocialCareReceived += careTransferred
            
            self.updateSocialCareNetworkSupply_W(receiver, supplier, 1)
            
            
#            if receiver == supplier:
#                self.inHouseInformalCare += self.p['quantumCare']
#                
#            household = list(supplier.occupants)
#            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
#            # If informal care is provided, it satisfies the most expensive cumulated child care need.
#            for member in householdCarers:
#                member.residualInformalSupply = member.residualInformalSupplies[distance]
#            householdCarers.sort(key=operator.attrgetter("residualInformalSupply"), reverse=True)
#            
#            if  receiver != supplier:
#                receiver.networkSupport += self.p['quantumCare']
#                
#            residualCare = self.p['quantumCare']
#            for i in householdCarers:
#                careForNeed = min(i.residualInformalSupply, residualCare)
#                if i in receiver.occupants and careForNeed > 0:
#                    i.careForFamily = True
#                i.socialWork += careForNeed
#                if careForNeed > 0:
#                    for j in range(4):
#                        i.residualInformalSupplies[j] -= careForNeed
#                        i.residualInformalSupplies[j] = max(i.residualInformalSupplies[j], 0)
#                    residualCare -= careForNeed
#                    if residualCare <= 0:
#                        break
#            
#            if receiver != supplier:
#                self.socialCareNetwork[receiver][supplier]['careTransferred'] += self.p['quantumCare']
#            else:
#                self.socialCareNetwork.node[receiver]['internalSupply'] += self.p['quantumCare']
#            
#            receiverOccupants = list(receiver.occupants)
#            careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
#            careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
#            residualCare = self.p['quantumCare']
#            for person in careTakers:
#                personalCare = min(person.unmetSocialCareNeed, residualCare)
#                person.unmetSocialCareNeed -= personalCare
#                person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
#                person.informalSocialCareReceived += personalCare
#                residualCare -= personalCare
#                if residualCare <= 0:
#                    break
#            receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
#            receiver.informalSocialCareReceived += self.p['quantumCare']
#            
#            self.updateSocialCareNetworkSupply_W(receiver, supplier, 1)
            
        else: 
        
            household = list(supplier.occupants)
            employed = [x for x in household if x.status == 'worker' and x.availableWorkingHours > 0 and x.maternityStatus == False]
            
            maxFormalCare = receiver.networkFormalSocialCareSupplies[indexSupplier]
            
            residualCare = min(self.p['quantumCare'], receiver.totalUnmetSocialCareNeed, maxFormalCare)
            if residualCare < 1.0:
                residualCare = 1.0
            
            costQuantumSocialCare = self.socialCareCost(supplier, residualCare)
            priceSocialCare = costQuantumSocialCare/residualCare
            
            if supplier.town != receiver.town or len(employed) == 0 or formalSocialCare == 0: # Only formal care is possible
                kindsOfCare = ['out-of-income care', 'out-of-wealth care']
                weights = [formalSocialCare, careFromWealth]
                careProbs = [x/sum(weights) for x in weights]
                care = np.random.choice(kindsOfCare, p = careProbs)
                if care == 'out-of-income care':
                    for j in range(4):
                        supplier.residualIncomeForSocialCare[j] -= costQuantumSocialCare
                        supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0.0)
                    supplier.householdFormalSupplyCost += costQuantumSocialCare
                    
                    careTransferred = 0
                    if len(employed) > 0:
                        employed.sort(key=operator.attrgetter("wage"), reverse=True)
                        residualCare = self.p['quantumCare']
                        for worker in employed:
                            maxCare = costQuantumSocialCare/worker.wage
                            workerCare = min(maxCare, residualCare)
                            if workerCare < 1.0:
                                workerCare = 1.0
                            worker.availableWorkingHours -= workerCare
                            worker.availableWorkingHours = float(max(int(worker.availableWorkingHours), 0))
                            worker.incomeExpenses += workerCare*worker.wage
                            careTransferred += workerCare
                            costQuantumSocialCare -= workerCare*worker.wage
                            residualCare -= workerCare
                            if residualCare <= 0:
                                break
                    
                    # print 'Post income for social care: ' + str(supplier.residualIncomeForSocialCare)
                    
                    receiverOccupants = list(receiver.occupants)                
                    careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
                    careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
                    residualCare = careTransferred
                    for person in careTakers:
                        personalCare = min(person.unmetSocialCareNeed, residualCare)
                        person.unmetSocialCareNeed -= personalCare
                        person.unmetSocialCareNeed = float(max(int(person.unmetSocialCareNeed), 0))
                        person.formalSocialCareReceived += personalCare
                        residualCare -= personalCare
                        if residualCare <= 0:
                            break
                    receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
                    receiver.formalSocialCareReceived += careTransferred
                    
                    self.updateSocialCareNetworkSupply_W(receiver, supplier, 2)
                
                
                
#                # Sample out-of-income/out-of-wealth according to quantity
#                kindsOfCare = ['out-of-income care', 'out-of-wealth care']
#                weights = [formalSocialCare, careFromWealth]
#                careProbs = [x/sum(weights) for x in weights]
#                care = np.random.choice(kindsOfCare, p = careProbs)
#                if care == 'out-of-income care':
#                    for j in range(4):
#                        supplier.residualIncomeForSocialCare[j] -= costQuantumSocialCare
#                        supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0.0)
#                    supplier.householdFormalSupplyCost += costQuantumSocialCare
#                    
#                    if len(employed) > 0:
#                        employed.sort(key=operator.attrgetter("wage"), reverse=True)
#                        residualCare = self.p['quantumCare']
#                        for worker in employed:
#                            maxCare = costQuantumSocialCare/worker.wage
#                            workerCare = min(maxCare, residualCare)
#                            worker.availableWorkingHours -= workerCare
#                            worker.availableWorkingHours = max(worker.availableWorkingHours, 0)
#                            costQuantumSocialCare -= workerCare*worker.wage
#                        
#                            worker.careExpenses += workerCare*worker.wage
#                            
#                            residualCare -= workerCare
#                            if residualCare <= 0:
#                                break
#                    
#                    # print 'Post income for social care: ' + str(supplier.residualIncomeForSocialCare)
#                    
#                    receiverOccupants = list(receiver.occupants)                
#                    careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
#                    careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
#                    residualCare = self.p['quantumCare']
#                    for person in careTakers:
#                        personalCare = min(person.unmetSocialCareNeed, residualCare)
#                        person.unmetSocialCareNeed -= personalCare
#                        person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
#                        person.formalSocialCareReceived += personalCare
#                        residualCare -= personalCare
#                        if residualCare <= 0:
#                            break
#                    receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
#                    receiver.formalSocialCareReceived += self.p['quantumCare']
#                    
#                    self.updateSocialCareNetworkSupply_W(receiver, supplier, 2)
                    
                else:
                    household = list(supplier.occupants)
                    householdCarers = [x for x in household if x.wealthForCare > 0]
                    householdCarers.sort(key=operator.attrgetter("wealthForCare"), reverse=True)
                    maxSupply = min(self.p['quantumCare'], sum([int(x.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])) for x in householdCarers]))
                    residualCare = maxSupply
                    careSupplied = 0
                    for i in householdCarers:
                        careForNeed = min(int(i.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])), residualCare)
                        if i in receiver.occupants and careForNeed > 0:
                            i.careForFamily = True
                        i.socialWork += careForNeed
                        if careForNeed > 0:
                            
                            i.wealthForCare -= careForNeed*self.p['priceSocialCare']
                            i.wealthForCare = max(i.wealthForCare, 0)
                            # i.financialWealth -= careForNeed*self.p['priceSocialCare']
                            # i.financialWealth = max(i.financialWealth, 0)
                            i.wealthSpentOnCare += careForNeed*self.p['priceSocialCare']
                            i.wealthPV = i.financialWealth*math.pow(1.0 + self.p['pensionReturnRate'], i.lifeExpectancy)
                            careSupplied += careForNeed
                            residualCare -= careForNeed
                            if residualCare <= 0:
                                break
                            
                    receiverOccupants = list(receiver.occupants)
                    careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
                    careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
                    residualCare = careSupplied
                    for person in careTakers:
                        personalCare = min(person.unmetSocialCareNeed, residualCare)
                        person.unmetSocialCareNeed -= personalCare
                        person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
                        person.formalSocialCareReceived += personalCare
                        residualCare -= personalCare
                        if residualCare <= 0:
                            break
                    receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
                    receiver.formalSocialCareReceived += self.p['quantumCare']
                    
                    self.updateSocialCareNetworkSupply_W(receiver, supplier, 10)
                    
#                for j in range(4):
#                    supplier.residualIncomeForSocialCare[j] -= costQuantumSocialCare
#                    supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0.0)
#                supplier.householdFormalSupplyCost += costQuantumSocialCare
#                
#                if len(employed) > 0:
#                    employed.sort(key=operator.attrgetter("wage"), reverse=True)
#                    residualCare = self.p['quantumCare']
#                    for worker in employed:
#                        maxCare = costQuantumSocialCare/worker.wage
#                        workerCare = min(maxCare, residualCare)
#                        worker.availableWorkingHours -= workerCare
#                        worker.availableWorkingHours = max(worker.availableWorkingHours, 0)
#                        costQuantumSocialCare -= workerCare*worker.wage
#                        residualCare -= workerCare
#                        if residualCare <= 0:
#                            break
#             
#                receiverOccupants = list(receiver.occupants)                
#                careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
#                careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
#                residualCare = self.p['quantumCare']
#                for person in careTakers:
#                    personalCare = min(person.unmetSocialCareNeed, residualCare)
#                    person.unmetSocialCareNeed -= personalCare
#                    person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
#                    person.formalSocialCareReceived += personalCare
#                    residualCare -= personalCare
#                    if residualCare <= 0:
#                        break
#                receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
#                receiver.formalSocialCareReceived += self.p['quantumCare']
#                
#                self.updateSocialCareNetworkSupply_W(receiver, supplier, 2)
                
            else:
                employed.sort(key=operator.attrgetter("wage"))
                
                carer = employed[0]
                if carer.wage > priceSocialCare: # In this case, it is more convenient to pay for formal care
                    
                    # Sample out-of-income/out-of-wealth according to quantity
                    kindsOfCare = ['out-of-income care', 'out-of-wealth care']
                    weights = [formalSocialCare, careFromWealth]
                    careProbs = [x/sum(weights) for x in weights]
                    care = np.random.choice(kindsOfCare, p = careProbs)
                    if care == 'out-of-income care':
                        
                        for j in range(4):
                            supplier.residualIncomeForSocialCare[j] -= costQuantumSocialCare
                            supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0.0)
                        supplier.householdFormalSupplyCost += costQuantumSocialCare
                        
                        careTransferred = 0
                        if len(employed) > 0:
                            employed.sort(key=operator.attrgetter("wage"), reverse=True)
                            residualCare = self.p['quantumCare']
                            for worker in employed:
                                maxCare = costQuantumSocialCare/worker.wage
                                workerCare = min(maxCare, residualCare)
                                if workerCare < 1.0:
                                    workerCare = 1.0
                                worker.availableWorkingHours -= workerCare
                                worker.availableWorkingHours = float(max(int(worker.availableWorkingHours), 0))
                                worker.incomeExpenses += workerCare*worker.wage
                                careTransferred += workerCare
                                costQuantumSocialCare -= workerCare*worker.wage
                                residualCare -= workerCare
                                if residualCare <= 0:
                                    break
                        
                        # print 'Post income for social care: ' + str(supplier.residualIncomeForSocialCare)
                        
                        receiverOccupants = list(receiver.occupants)                
                        careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
                        careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
                        residualCare = careTransferred
                        for person in careTakers:
                            personalCare = min(person.unmetSocialCareNeed, residualCare)
                            person.unmetSocialCareNeed -= personalCare
                            person.unmetSocialCareNeed = float(max(int(person.unmetSocialCareNeed), 0))
                            person.formalSocialCareReceived += personalCare
                            residualCare -= personalCare
                            if residualCare <= 0:
                                break
                        receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
                        receiver.formalSocialCareReceived += careTransferred
                        
                        self.updateSocialCareNetworkSupply_W(receiver, supplier, 2)
                        
                        
                    
#                        for j in range(4):
#                            supplier.residualIncomeForSocialCare[j] -= costQuantumSocialCare
#                            supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0.0)
#                        supplier.householdFormalSupplyCost += costQuantumSocialCare
#                        
#                        if len(employed) > 0:
#                            employed.sort(key=operator.attrgetter("wage"), reverse=True)
#                            residualCare = self.p['quantumCare']
#                            for worker in employed:
#                                maxCare = costQuantumSocialCare/worker.wage
#                                workerCare = min(maxCare, residualCare)
#                                worker.availableWorkingHours -= workerCare
#                                worker.availableWorkingHours = max(worker.availableWorkingHours, 0)
#                                costQuantumSocialCare -= workerCare*worker.wage
#
#                                worker.careExpenses += workerCare*worker.wage
#                            
#                                residualCare -= workerCare
#                                if residualCare <= 0:
#                                    break
#                        
#                        # print 'Post income for social care: ' + str(supplier.residualIncomeForSocialCare)
#                        
#                        receiverOccupants = list(receiver.occupants)                
#                        careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
#                        careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
#                        residualCare = self.p['quantumCare']
#                        for person in careTakers:
#                            personalCare = min(person.unmetSocialCareNeed, residualCare)
#                            person.unmetSocialCareNeed -= personalCare
#                            person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
#                            person.formalSocialCareReceived += personalCare
#                            residualCare -= personalCare
#                            if residualCare <= 0:
#                                break
#                        receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
#                        receiver.formalSocialCareReceived += self.p['quantumCare']
#                        
#                        self.updateSocialCareNetworkSupply_W(receiver, supplier, 2)
                        
                    else:
                        household = list(supplier.occupants)
                        householdCarers = [x for x in household if x.wealthForCare > 0]
                        householdCarers.sort(key=operator.attrgetter("wealthForCare"), reverse=True)
                        maxSupply = min(self.p['quantumCare'], sum([int(x.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])) for x in householdCarers]))
                        residualCare = maxSupply
                        careSupplied = 0
                        for i in householdCarers:
                            careForNeed = min(int(i.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])), residualCare)
                            if i in receiver.occupants and careForNeed > 0:
                                i.careForFamily = True
                            i.socialWork += careForNeed
                            if careForNeed > 0:
                                
                                i.wealthForCare -= careForNeed*self.p['priceSocialCare']
                                i.wealthForCare = max(i.wealthForCare, 0)
                                # i.financialWealth -= careForNeed*self.p['priceSocialCare']
                                # i.financialWealth = max(i.financialWealth, 0)
                                i.wealthSpentOnCare += careForNeed*self.p['priceSocialCare']
                                i.wealthPV = i.financialWealth*math.pow(1.0 + self.p['pensionReturnRate'], i.lifeExpectancy)
                                careSupplied += careForNeed
                                residualCare -= careForNeed
                                if residualCare <= 0:
                                    break
                                
                        receiverOccupants = list(receiver.occupants)
                        careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
                        careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
                        residualCare = careSupplied
                        for person in careTakers:
                            personalCare = min(person.unmetSocialCareNeed, residualCare)
                            person.unmetSocialCareNeed -= personalCare
                            person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
                            person.formalSocialCareReceived += personalCare
                            residualCare -= personalCare
                            if residualCare <= 0:
                                break
                        receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
                        receiver.formalSocialCareReceived += self.p['quantumCare']
                        
                        self.updateSocialCareNetworkSupply_W(receiver, supplier, 10)
                    
                    
                        
                        

#                    for j in range(4):
#                        supplier.residualIncomeForSocialCare[j] -= costQuantumSocialCare
#                        supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0)
#                    supplier.householdFormalSupplyCost += costQuantumSocialCare
#                    
#                    employed.sort(key=operator.attrgetter("wage"), reverse=True)
#                    residualCare = self.p['quantumCare']
#                    for worker in employed:
#                        maxCare = costQuantumSocialCare/worker.wage
#                        workerCare = min(maxCare, residualCare)
#                        worker.availableWorkingHours -= workerCare
#                        worker.availableWorkingHours = max(worker.availableWorkingHours, 0)
#                        costQuantumSocialCare -= workerCare*worker.wage
#                        residualCare -= workerCare
#                        if residualCare <= 0:
#                            break
#                    
#                    receiverOccupants = list(receiver.occupants)
#                    careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
#                    careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
#                    residualCare = self.p['quantumCare']
#                    for person in careTakers:
#                        personalCare = min(person.unmetSocialCareNeed, residualCare)
#                        person.unmetSocialCareNeed -= personalCare
#                        person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
#                        person.formalSocialCareReceived += personalCare
#                        residualCare -= personalCare
#                        if residualCare <= 0:
#                            break
#                    receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
#                    receiver.formalSocialCareReceived += self.p['quantumCare']
#                    
#                    self.updateSocialCareNetworkSupply_W(receiver, supplier, 3)
                    
                else: # In this case, it is more convenient to take time off work to provide informal care: out-of-income
#                    print 'Case 4'
#                    print 'Carer pre available working hours: ' + str(carer.availableWorkingHours)
                    
                    if receiver == supplier:
                        self.inHouseInformalCare += self.p['quantumCare']
                    
                    careTransferred = min(self.p['quantumCare'], carer.availableWorkingHours)
                    if careTransferred == 0:
                        print 'No residual working hours in carer'
                        sys.exit()
                    if carer in receiver.occupants and careTransferred > 0:
                        carer.careForFamily = True
                    incomeForCare = careTransferred*carer.wage
                    carer.availableWorkingHours -= careTransferred #self.p['quantumCare']
                    carer.availableWorkingHours = max(0, carer.availableWorkingHours)
                    carer.residualWorkingHours -= careTransferred
                    carer.residualWorkingHours = max(0, carer.residualWorkingHours)
                    carer.outOfWorkSocialCare += careTransferred
                    carer.socialWork += careTransferred
                    
                    
                    if receiver != supplier:
                        self.socialCareNetwork[receiver][supplier]['careTransferred'] += careTransferred
                    else:
                        self.socialCareNetwork.node[receiver]['internalSupply'] += careTransferred
#                    print 'Care transferred: ' + str(careTransferred)
#                    print 'Carer wage: ' + str(carer.wage)
#                    print 'Carer post available working hours: ' + str(carer.availableWorkingHours)
#                    print 'Pre-residual Income for care (4): ' + str(supplier.residualIncomeForSocialCare)
                    
                    for j in range(4):
                        supplier.residualIncomeForSocialCare[j] -= incomeForCare
                        supplier.residualIncomeForSocialCare[j] = max(supplier.residualIncomeForSocialCare[j], 0)
                        
#                    print 'Post-residual Income for care (4): ' + str(supplier.residualIncomeForSocialCare)
                    
                    receiverOccupants = list(receiver.occupants)
                    careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
                    careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
                    residualCare = careTransferred
                    for person in careTakers:
                        personalCare = min(person.unmetSocialCareNeed, residualCare)
                        person.unmetSocialCareNeed -= personalCare
                        person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
                        person.informalSocialCareReceived += personalCare
                        residualCare -= personalCare
                        if residualCare <= 0:
                            break
                    receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
                    receiver.informalSocialCareReceived += self.p['quantumCare']
                    
                    self.updateSocialCareNetworkSupply_W(receiver, supplier, 4)
                    
#        else: # Out-of-wealth care
#            household = list(supplier.occupants)
#            householdCarers = [x for x in household if x.wealthForCare > 0]
#            householdCarers.sort(key=operator.attrgetter("wealthForCare"), reverse=True)
#            maxSupply = min(self.p['quantumCare'], sum([int(x.wealthForCare/self.p['priceSocialCare']) for x in householdCarers]))
#            residualCare = maxSupply
#            careSupplied = 0
#            for i in householdCarers:
#                careForNeed = min(int(i.wealthForCare/self.p['priceSocialCare']), residualCare)
#                if i in receiver.occupants and careForNeed > 0:
#                    i.careForFamily = True
#                i.socialWork += careForNeed
#                if careForNeed > 0:
#                    
#                    i.wealthForCare -= careForNeed*self.p['priceSocialCare']
#                    i.wealthForCare = max(i.wealthForCare, 0)
#                    i.financialWealth -= careForNeed*self.p['priceSocialCare']
#                    i.financialWealth = max(i.financialWealth, 0)
#                    i.wealthSpentOnCare += careForNeed*self.p['priceSocialCare']
#                    i.wealthPV = i.financialWealth*math.pow(1.0 + self.p['pensionReturnRate'], i.lifeExpectancy)
#                    careSupplied += careForNeed
#
#                    residualCare -= careForNeed
#                    if residualCare <= 0:
#                        break
#                    
#            receiverOccupants = list(receiver.occupants)
#            careTakers = [x for x in receiverOccupants if x.unmetSocialCareNeed > 0]
#            careTakers.sort(key=operator.attrgetter("unmetSocialCareNeed"), reverse=True)
#            residualCare = careSupplied
#            for person in careTakers:
#                personalCare = min(person.unmetSocialCareNeed, residualCare)
#                person.unmetSocialCareNeed -= personalCare
#                person.unmetSocialCareNeed = max(person.unmetSocialCareNeed, 0)
#                person.formalSocialCareReceived += personalCare
#                residualCare -= personalCare
#                if residualCare <= 0:
#                    break
#            receiver.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in careTakers])
#            receiver.formalSocialCareReceived += self.p['quantumCare']
#            
#            self.updateSocialCareNetworkSupply_W(receiver, supplier, 10)
                    
    def transferSocialCare_Ind(self, receiver, supplier):
        
        if receiver != supplier:
            # In this case, care is provided with resources from people other than the care receiver
            distance = receiver.careNetwork[receiver][supplier]['distance']
            indexSupplier = receiver.suppliers.index(supplier)
            informalSupply = receiver.networkInformalSupplies[indexSupplier]
            formalSocialCare = receiver.networkFormalSocialCareSupplies[indexSupplier]

            informalFactor = math.pow(informalSupply, self.p['betaInformalCare'])
            formalFactor = math.pow(formalSocialCare, self.p['betaFormalCare'])
            probInformalCare = informalFactor/(informalFactor+formalFactor)
            # Select kind of care based on supplier availability
            care = 'formal care'
            if supplier.house.town == receiver.house.town:
                # careWeights = [informalSupply, formalSocialCare, careFromWealth]
                # careProbs = [x/sum(careWeights) for x in careWeights]
                careProbs = [probInformalCare, 1-probInformalCare]
                care = np.random.choice(['informal care', 'formal care'], p = careProbs) 
            
            if care == 'informal care':
            
                household = list(supplier.house.occupants)
                householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
                # If informal care is provided, it satisfies the most expensive cumulated child care need.
                for member in householdCarers:
                    member.residualInformalSupply = member.residualInformalSupplies[distance]
                householdCarers.sort(key=operator.attrgetter("residualInformalSupply"), reverse=True)
                
                residualCare = min(self.p['quantumCare'], receiver.unmetSocialCareNeed)
                if residualCare < 1.0:
                    residualCare = 1.0
                
                careTransferred = 0
                for i in householdCarers:
                    careForNeed = min(i.residualInformalSupply, residualCare)
                    if careForNeed < 1.0:
                        careForNeed = 1.0
                    if i in receiver.house.occupants and careForNeed > 0:
                        i.careForFamily = True
                    i.socialWork += careForNeed
                    
                    ## Record contact details for exposure process
                    if receiver.house.id != supplier.house.id: 
                        # Add contact details to supplier care contact list
                        if len([x for x in i.nokCareContacts if x.contact == receiver]) == 0:
                            newContact = CareContact(receiver, careForNeed)
                            i.nokCareContacts.append(newContact)
                        else:
                            careContact = [x for x in i.nokCareContacts if x.contact == receiver][0]
                            careContact.contactDuration += careForNeed
                        # Add contact details to receiver care contact list
                        if len([x for x in receiver.nokCareContacts if x.contact == i]) == 0:
                            newContact = CareContact(i, careForNeed)
                            receiver.nokCareContacts.append(newContact)
                        else:
                            careContact = [x for x in receiver.nokCareContacts if x.contact == i][0]
                            careContact.contactDuration += careForNeed
                    ########################################################################################
                            
                    if careForNeed > 0:
                        for j in range(4):
                            i.residualInformalSupplies[j] -= careForNeed
                            i.residualInformalSupplies[j] = float(max(int(i.residualInformalSupplies[j]), 0))
                        careTransferred += careForNeed
                        residualCare -= careForNeed
                        if residualCare <= 0:
                            break
                        
                if receiver.house.id == supplier.house.id:
                    self.inHouseInformalCare += careTransferred
                    
                if  receiver.house.id != supplier.house.id:
                    receiver.house.networkSupport += careTransferred
                    
                if receiver != supplier:
                    self.socialCareNetwork[receiver][supplier]['careTransferred'] += careTransferred
                else:
                    self.socialCareNetwork.node[receiver]['internalSupply'] += careTransferred

                receiver.unmetSocialCareNeed -= careTransferred
                receiver.unmetSocialCareNeed = float(max(int(receiver.unmetSocialCareNeed), 0))
                receiver.informalSocialCareReceived += careTransferred
                receiver.house.informalSocialCareReceived += careTransferred
                
                self.updateSocialCareNetworkSupply_Ind(receiver, supplier, 1)
                
            else: # Out-of-income care. Could be formal or informal: if different town, only formal; otherwise, depends on lowest wage
                household = list(supplier.house.occupants)
                employed = [x for x in household if x.status == 'worker' and x.availableWorkingHours > 0 and x.maternityStatus == False]
                
                maxFormalCare = receiver.networkFormalSocialCareSupplies[indexSupplier]
                
                residualCare = min(self.p['quantumCare'], receiver.unmetSocialCareNeed, maxFormalCare)
                if residualCare < 1.0:
                    residualCare = 1.0
                
                costQuantumSocialCare = self.socialCareCost(supplier, residualCare)
                priceSocialCare = costQuantumSocialCare/residualCare
                
                if supplier.house.town != receiver.house.town or len(employed) == 0: # Only formal care is possible
                    for j in range(4):
                        supplier.house.residualIncomeForSocialCare[j] -= costQuantumSocialCare
                        supplier.house.residualIncomeForSocialCare[j] = max(supplier.house.residualIncomeForSocialCare[j], 0.0)
                    supplier.house.householdFormalSupplyCost += costQuantumSocialCare
                    
                    careTransferred = residualCare
                    if len(employed) > 0:
                        careTransferred = 0
                        employed.sort(key=operator.attrgetter("wage"), reverse=True)
                        for worker in employed:
                            maxCare = costQuantumSocialCare/worker.wage
                            workerCare = min(maxCare, residualCare)
                            if workerCare < 1.0:
                                workerCare = 1.0
                            worker.availableWorkingHours -= workerCare
                            worker.availableWorkingHours = float(max(int(worker.availableWorkingHours), 0))
                            worker.incomeExpenses += workerCare*worker.wage
                            careTransferred += workerCare
                            costQuantumSocialCare -= workerCare*worker.wage
                            residualCare -= workerCare
                            if residualCare <= 0:
                                break
                    
                    # print 'Post income for social care: ' + str(supplier.residualIncomeForSocialCare)
                    receiver.unmetSocialCareNeed -= careTransferred
                    receiver.unmetSocialCareNeed = float(max(int(receiver.unmetSocialCareNeed), 0))
                    receiver.formalSocialCareReceived += careTransferred
                    receiver.house.formalSocialCareReceived += careTransferred
                    
                    self.updateSocialCareNetworkSupply_Ind(receiver, supplier, 2)
                    
                else:
                    # Select the worker with the lowest pay
                    employed.sort(key=operator.attrgetter("wage"))
                    
                    carer = employed[0]
                    if carer.wage > priceSocialCare: # In this case, it is more convenient to pay for formal care
                        for j in range(4):
                            supplier.house.residualIncomeForSocialCare[j] -= costQuantumSocialCare
                            supplier.house.residualIncomeForSocialCare[j] = max(supplier.house.residualIncomeForSocialCare[j], 0)
                        supplier.house.householdFormalSupplyCost += costQuantumSocialCare
                        
                        # Workers working hours must be commited to earn salary to buy formal care, so are not available for informal care.
                        employed.sort(key=operator.attrgetter("wage"), reverse=True)
                        careToAllocate = residualCare
                        careTransferred = 0
                        for worker in employed:
                            maxCare = costQuantumSocialCare/worker.wage
                            workerCare = min(maxCare, careToAllocate)
                            if workerCare < 1.0:
                                workerCare = 1.0
                            careTransferred += workerCare
                            worker.availableWorkingHours -= workerCare
                            worker.availableWorkingHours = float(max(int(worker.availableWorkingHours), 0))
                            worker.incomeExpenses += workerCare*worker.wage
                            costQuantumSocialCare -= workerCare*worker.wage
                            careToAllocate -= workerCare
                            if careToAllocate <= 0:
                                break
                       
                        receiver.unmetSocialCareNeed -= careTransferred
                        receiver.unmetSocialCareNeed = float(max(int(receiver.unmetSocialCareNeed), 0))
                        receiver.formalSocialCareReceived += careTransferred
                        receiver.house.formalSocialCareReceived += careTransferred
                        
                        self.updateSocialCareNetworkSupply_Ind(receiver, supplier, 3)
                        
                    else: # In this case, it is more convenient to take time off work to provide informal care.
#                        print 'Case 4'
#                        print 'Carer pre available working hours: ' + str(carer.availableWorkingHours)
                        
                        
                        
                        careTransferred = min(self.p['quantumCare'], carer.availableWorkingHours, receiver.unmetSocialCareNeed)
                        if careTransferred == 0:
                            print 'No residual working hours in carer'
                            sys.exit()
                        if careTransferred < 1.0:
                            careTransferred = 1.0
                        if carer in receiver.house.occupants and careTransferred > 0:
                            carer.careForFamily = True
                        incomeForCare = careTransferred*carer.wage
                        carer.availableWorkingHours -= careTransferred #self.p['quantumCare']
                        carer.availableWorkingHours = float(max(0, int(carer.availableWorkingHours)))
                        carer.residualWorkingHours -= careTransferred
                        carer.residualWorkingHours = float(max(0, int(carer.residualWorkingHours)))
                        carer.outOfWorkSocialCare += careTransferred
                        carer.socialWork += careTransferred
                        carer.house.outOfWorkSocialCare += careTransferred
                        
                        ## Record contact details for exposure process
                        if receiver.house.id != carer.house.id: 
                            # Add contact details to supplier care contact list
                            if len([x for x in carer.nokCareContacts if x.contact == receiver]) == 0:
                                newContact = CareContact(receiver, careForNeed)
                                carer.nokCareContacts.append(newContact)
                            else:
                                careContact = [x for x in carer.nokCareContacts if x.contact == receiver][0]
                                careContact.contactDuration += careForNeed
                            # Add contact details to receiver care contact list
                            if len([x for x in receiver.nokCareContacts if x.contact == carer]) == 0:
                                newContact = CareContact(carer, careForNeed)
                                receiver.nokCareContacts.append(newContact)
                            else:
                                careContact = [x for x in receiver.nokCareContacts if x.contact == carer][0]
                                careContact.contactDuration += careForNeed
                        ########################################################################################
                        
                        if receiver.house.id == supplier.house.id:
                            self.inHouseInformalCare += careTransferred
                        
#                        print 'Care transferred: ' + str(careTransferred)
#                        print 'Carer wage: ' + str(carer.wage)
#                        print 'Carer post available working hours: ' + str(carer.availableWorkingHours)
#                        print 'Pre-residual Income for care (4): ' + str(supplier.house.residualIncomeForSocialCare)
                        
                        for j in range(4):
                            supplier.house.residualIncomeForSocialCare[j] -= incomeForCare
                            supplier.house.residualIncomeForSocialCare[j] = max(supplier.house.residualIncomeForSocialCare[j], 0)
                            
                        # print 'Post-residual Income for care (4): ' + str(supplier.house.residualIncomeForSocialCare)
                        
                        receiver.unmetSocialCareNeed -= careTransferred
                        receiver.unmetSocialCareNeed = float(max(int(receiver.unmetSocialCareNeed), 0))
                        receiver.informalSocialCareReceived += careTransferred
                        receiver.house.informalSocialCareReceived += careTransferred
                        
                        self.updateSocialCareNetworkSupply_Ind(receiver, supplier, 4)
                
                
        else: # In this case, formal care is paid with the receiver's wealth
            careSupplied = min(receiver.careSupplyFromWealth, self.p['quantumCare'], receiver.unmetSocialCareNeed)
            receiver.wealthForCare -= careSupplied*self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])
            receiver.wealthForCare = max(receiver.wealthForCare, 0)
            receiver.wealthSpentOnCare += careSupplied*self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])

            receiver.unmetSocialCareNeed -= careSupplied
            receiver.unmetSocialCareNeed = max(receiver.unmetSocialCareNeed, 0)
            receiver.formalSocialCareReceived += careSupplied
            receiver.house.formalSocialCareReceived += careSupplied
            self.updateSocialCareNetworkSupply_Ind(receiver, supplier, 10)
            
        
    def socialCareCost(self, person, care):
        house = person.house
        availableIncomeByTaxBand = self.updateIncomeByTaxBand(house)
        prices = [self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])*(1.0-x*self.p['taxBreakRate']) for x in self.p['taxationRates']]
        cost = 0
        residualCare = care
        for i in range(len(availableIncomeByTaxBand)):
            # house.incomeByTaxBand[i] needs to be updated at every transfer of formal care, net of time off work and total formal care cost
            if availableIncomeByTaxBand[i]/prices[i] > residualCare:
                cost += residualCare*prices[i]
                availableIncomeByTaxBand[i] -= residualCare*prices[i]
                break
            else:
                cost += availableIncomeByTaxBand[i]
                residualCare -= availableIncomeByTaxBand[i]/prices[i]
                availableIncomeByTaxBand[i] = 0
        return cost    
        
        
    def computeResidualIncomeForSocialCare(self):
        
        # incomeShares = []
        
        for house in self.map.occupiedHouses:
            
            household = list(house.occupants)
            householdCarers = householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            employed = [x for x in householdCarers if x.status == 'worker']
            
            employed = [x for x in householdCarers if x.status == 'worker']
            for worker in employed:
                worker.potentialIncome = worker.residualWorkingHours*worker.wage
                
            incomes = [x.potentialIncome for x in householdCarers]
            incomes.extend([x.income for x in household if x.status == 'retired'])
            netIncome = sum(incomes) - house.householdFormalSupplyCost
            netIncome = max(netIncome, 0)
            incomePerCapita = netIncome/float(len(household))
            
            house.householdInformalSupply = []
            for i in range(4):
                house.householdInformalSupply.append(sum([x.residualInformalSupplies[i] for x in householdCarers]))
    
            taxBands = len(self.p['taxBrackets'])
            house.incomeByTaxBand = [0]*taxBands
            house.incomeByTaxBand[-1] = sum(incomes)
            for i in range(taxBands-1):
                for income in incomes:
                    if income > self.p['taxBrackets'][i]:
                        bracket = income-self.p['taxBrackets'][i]
                        house.incomeByTaxBand[i] += bracket
                        house.incomeByTaxBand[-1] -= bracket
                        incomes[incomes.index(income)] -= bracket
            
#            ############  Check Variable  ############################
#            if self.year == self.p['getCheckVariablesAtYear']:
#                self.perCapitaHouseholdIncome.append(incomePerCapita)
#            ##########################################################
     
            incomeForCareShare_D0 = 1.0 - 1.0/math.exp(self.p['incomeCareParam']*incomePerCapita)
            
            # incomeShares.append(incomeForCareShare_D0)
            
            incomeForCareShare_D1 = (1.0 - 1.0/math.exp(self.p['incomeCareParam']*incomePerCapita))*self.p['formalCareDiscountFactor']
            
            # print incomeForCareShare_D1
            
            residualIncomeForCare_D0 = netIncome*incomeForCareShare_D0
            residualIncomeForCare_D1 = netIncome*incomeForCareShare_D1
            
            house.residualIncomeForSocialCare = [residualIncomeForCare_D0, residualIncomeForCare_D1, 0, 0]
            
            # Compute total supply of household
            house.totalSupplies = [self.updateFormalSocialCareSupplies(house, x) for x in [0, 1]]
            house.totalSupplies.extend([0,0])
            
            for j in range(4):
                house.totalSupplies[j] += sum([x.residualInformalSupplies[j] for x in house.occupants])
                
            # Add supply from wealth
            peopleWithNeed = [x for x in house.occupants if x.wealthForCare > 0]
            house.totalLifeExpectancy = sum([x.lifeExpectancy for x in peopleWithNeed])
            house.careSupplyFromWealth = 0
            if house.totalLifeExpectancy > 0:
                house.wealthForCare = sum([x.wealthForCare for x in peopleWithNeed])
                # weeklyWealth = sum([x.wealthForCare for x in peopleWithNeed]) # /float(52*house.totalLifeExpectancy)
                house.totalSupplies[0] += float(int(house.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])))
                
            house.netCareSupply = house.totalSupplies[0] - house.totalSocialCareNeed
            
            
    
    def computeSocialCareNetworkSupply(self, house):
        
        town = house.town
        house.networkSupply = 0
        house.networkTotalSupplies = []
        house.weightedTotalSupplies = []
        house.networkInformalSupplies = []
        house.networkFormalSocialCareSupplies = []
        
        # Care from wealth
        peopleWithNeed = [x for x in house.occupants if x.wealthForCare > 0]
        house.totalLifeExpectancy = sum([x.lifeExpectancy for x in peopleWithNeed])
        house.careSupplyFromWealth = 0
        if house.totalLifeExpectancy > 0:
            house.wealthForCare = sum([x.wealthForCare for x in peopleWithNeed])
            weeklyWealth = sum([x.financialWealth for x in peopleWithNeed])/float(52*house.totalLifeExpectancy)
            house.careSupplyFromWealth = float(int(weeklyWealth/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])))
            
        house.networkSupply += house.careSupplyFromWealth
        
        house.suppliers = [house]
        house.suppliers.extend(list(house.careNetwork.neighbors(house)))
        
        for supplier in house.suppliers:
            distance = 0
            if supplier != house:
                distance = house.careNetwork[house][supplier]['distance']
                
#            careFromWealth = 0
#            if supplier == house:
#                careFromWealth = house.careSupplyFromWealth
#            weightedCareFromWealth = careFromWealth*self.p['weightCareFromWealth']
            
            # Informal Care Supplies
            household = list(supplier.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            # householdInformalSupply = []
            # for i in range(4):
                # householdInformalSupply.append(sum([x.residualInformalSupplies[i] for x in householdCarers]))
            informalSupply = 0
            if supplier.town == town:
                informalSupply = float(sum([x.residualInformalSupplies[distance] for x in householdCarers]))
            house.networkInformalSupplies.append(informalSupply)
            house.networkSupply += informalSupply
            weightedInformalSupply = informalSupply
            
            # Formal Social Care supply
            maxFormalSocialCare = float(self.updateFormalSocialCareSupplies(supplier, distance))
            house.networkFormalSocialCareSupplies.append(maxFormalSocialCare)
            weightedMaxFormalSocialCare = maxFormalSocialCare
            
            house.networkTotalSupplies.append(informalSupply+maxFormalSocialCare)
            house.weightedTotalSupplies.append(weightedInformalSupply+weightedMaxFormalSocialCare)
            house.networkSupply += maxFormalSocialCare
            
    def computeSocialCareNetworkSupply_W(self, house):
        
        town = house.town
        house.networkSupply = 0
        house.networkTotalSupplies = []
        house.weightedTotalSupplies = []
        house.networkInformalSupplies = []
        house.networkFormalSocialCareSupplies = []
        # Care from wealth
        peopleWithNeed = [x for x in house.occupants if x.wealthForCare > 0]
        house.totalLifeExpectancy = sum([x.lifeExpectancy for x in peopleWithNeed])
        house.careSupplyFromWealth = 0
        if house.totalLifeExpectancy > 0:
            house.wealthForCare = sum([x.wealthForCare for x in peopleWithNeed])
            # weeklyWealth = sum([x.wealthForCare for x in peopleWithNeed])/float(52*house.totalLifeExpectancy)
            house.careSupplyFromWealth = float(int(house.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])))
            
        house.networkSupply += house.careSupplyFromWealth
        
        house.suppliers = [house]
        house.suppliers.extend(list(house.careNetwork.neighbors(house)))
        
        for supplier in house.suppliers:
            distance = 0
            if supplier != house:
                distance = house.careNetwork[house][supplier]['distance']
                
            careFromWealth = 0
            if supplier == house:
                careFromWealth = house.careSupplyFromWealth
            weightedCareFromWealth = careFromWealth #*self.p['weightCareFromWealth']
            
            # Informal Care Supplies
            household = list(supplier.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            # householdInformalSupply = []
            # for i in range(4):
                # householdInformalSupply.append(sum([x.residualInformalSupplies[i] for x in householdCarers]))
            informalSupply = 0
            if supplier.town == town:
                informalSupply = float(sum([x.residualInformalSupplies[distance] for x in householdCarers]))
            house.networkInformalSupplies.append(informalSupply)
            house.networkSupply += informalSupply
            weightedInformalSupply = informalSupply
            
            # Formal Social Care supply
            maxFormalSocialCare = float(self.updateFormalSocialCareSupplies(supplier, distance))
            house.networkFormalSocialCareSupplies.append(maxFormalSocialCare)
            weightedMaxFormalSocialCare = maxFormalSocialCare #*self.p['weightCareFromIncome']
            
            house.networkTotalSupplies.append(informalSupply+maxFormalSocialCare+careFromWealth)
            house.weightedTotalSupplies.append(weightedInformalSupply+weightedMaxFormalSocialCare+weightedCareFromWealth)
            house.networkSupply += maxFormalSocialCare
            
    def computeSocialCareNetworkSupply_Ind(self, person):
        
        town = person.house.town
        person.networkSupply = 0
        person.networkTotalSupplies = []
        person.weightedTotalSupplies = []
        person.networkInformalSupplies = []
        person.networkFormalSocialCareSupplies = []
        
        person.suppliers = list(person.careNetwork.neighbors(person))
        
#        print person.id
#        print [x.id for x in person.suppliers]
        
        person.careSupplyFromWealth = float(int(person.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])))
        careFromWealth = person.careSupplyFromWealth
        # weightedCareFromWealth = careFromWealth*self.p['weightCareFromWealth']
        person.networkSupply += person.careSupplyFromWealth
        
        for supplier in person.suppliers:
            
            supplierTown = supplier.house.town
            distance = person.careNetwork[person][supplier]['distance']
                
            # Informal Care Supplies
            household = list(supplier.house.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            # householdInformalSupply = []
            # for i in range(4):
                # householdInformalSupply.append(sum([x.residualInformalSupplies[i] for x in householdCarers]))
            
            informalSupply = 0
            if supplierTown == town:
                informalSupply = float(sum([x.residualInformalSupplies[distance] for x in householdCarers]))
                if supplier.house == person.house:
                    person.inHouseInformalCare += float(sum([x.residualInformalSupplies[distance] for x in householdCarers]))
                else:
                    person.externalInformalCare += float(sum([x.residualInformalSupplies[distance] for x in householdCarers]))
                    
            person.networkInformalSupplies.append(informalSupply)
            person.networkSupply += informalSupply
            weightedInformalSupply = informalSupply
            
            # Formal Social Care supply
            maxFormalSocialCare = float(self.updateFormalSocialCareSupplies_Ind(supplier, distance))
            person.networkFormalSocialCareSupplies.append(maxFormalSocialCare)
            
            informalFactor = math.pow(informalSupply, self.p['betaInformalCare'])
            formalFactor = math.pow(maxFormalSocialCare, self.p['betaFormalCare'])
            
            # weightedMaxFormalSocialCare = maxFormalSocialCare*self.p['weightCareFromIncome']
            
            person.networkTotalSupplies.append(informalSupply+maxFormalSocialCare)
            person.weightedTotalSupplies.append(informalFactor+formalFactor)
            person.networkSupply += maxFormalSocialCare
        
    def updateSocialCareNetworkSupply(self, house, supplier, n):
        
        town = house.town
        
        oldSupply = house.networkSupply
        oldInformalSupplies = list(house.networkInformalSupplies)
        oldFormalCareSupplies = list(house.networkFormalSocialCareSupplies)
        oldTotalSupplies = list(house.networkTotalSupplies)
        oldWealthForCare = house.wealthForCare
        oldCareFromWealth = house.careSupplyFromWealth
        
        peopleWithNeed = [x for x in house.occupants if x.wealthForCare > 0]
        
#        if n == 10:
#            print 'Wealth for care; ' + str([x.wealthForCare for x in peopleWithNeed])
        
        # totalLifeExpectancy = sum([x.lifeExpectancy for x in peopleWithNeed])
        house.careSupplyFromWealth = 0
        if house.totalLifeExpectancy > 0:
            house.wealthForCare = sum([x.wealthForCare for x in peopleWithNeed])
            weeklyWealth = sum([x.financialWealth for x in peopleWithNeed])/float(356*house.totalLifeExpectancy)
            house.careSupplyFromWealth = float(int(weeklyWealth/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])))
            
#            if n == 10:
#                print 'House life expectancy: ' + str(house.totalLifeExpectancy)
#                print 'Weekly wealth: ' + str(weeklyWealth)
#                print 'Care supply from wealth: ' + str(house.careSupplyFromWealth)
#                print 'Old wealth for care: ' + str(oldWealthForCare)
#                print 'New wealth for care: ' + str(house.wealthForCare)
                
        
        if supplier in house.suppliers:
        
            supplierIndex = house.suppliers.index(supplier)
            
#            careFromWealth = 0
#            if supplier == house:
#                careFromWealth = house.careSupplyFromWealth
#            weightedCareFromWealth = careFromWealth*self.p['weightCareFromWealth']
        
            distance = 0
            if supplier != house:
                distance = house.careNetwork[house][supplier]['distance']
                
            household = list(supplier.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            householdInformalSupply = []
            for i in range(4):
                householdInformalSupply.append(sum([x.residualInformalSupplies[i] for x in householdCarers]))
            informalSupply = 0
            if supplier.town == town:
                informalSupply = float(sum([x.residualInformalSupplies[distance] for x in householdCarers]))
            house.networkInformalSupplies[supplierIndex] = informalSupply
            weightedInformalSupply = informalSupply
            
            # print 'Updating formal care supply...'
            maxFormalSocialCare = float(self.updateFormalSocialCareSupplies(supplier, distance))
            weightedMaxFormalSocialCare = maxFormalSocialCare
    
            house.networkFormalSocialCareSupplies[supplierIndex] = maxFormalSocialCare
            
            house.networkTotalSupplies[supplierIndex] = informalSupply+maxFormalSocialCare
            house.weightedTotalSupplies[supplierIndex] = weightedInformalSupply+weightedMaxFormalSocialCare
            
            house.networkSupply = sum(house.networkTotalSupplies)
            
            if house.networkSupply >= oldSupply and n != 0 and n != 10:
                print 'Error: social care supply did not decrease!'
                print 'Case: ' + str(n)
                print house.id
                print [x.id for x in house.suppliers]
                print supplierIndex
                print supplier.id
                print ''
                print oldSupply
                print house.networkSupply
                print ''
                print oldInformalSupplies
                print house.networkInformalSupplies
                print ''
                print oldFormalCareSupplies
                print house.networkFormalSocialCareSupplies
                print ''
                print oldCareFromWealth
                print house.careSupplyFromWealth
                print ''
                print oldTotalSupplies
                print house.networkTotalSupplies
                
                sys.exit()
    
    def updateSocialCareNetworkSupply_W(self, house, supplier, n):
        
        town = house.town
        
        oldSupply = house.networkSupply
        oldInformalSupplies = list(house.networkInformalSupplies)
        oldFormalCareSupplies = list(house.networkFormalSocialCareSupplies)
        oldTotalSupplies = list(house.networkTotalSupplies)
        oldWealthForCare = house.wealthForCare
        oldCareFromWealth = house.careSupplyFromWealth
        
        peopleWithNeed = [x for x in house.occupants if x.wealthForCare > 0]
        
#        if n == 10:
#            print 'Wealth for care; ' + str([x.wealthForCare for x in peopleWithNeed])
        
        # totalLifeExpectancy = sum([x.lifeExpectancy for x in peopleWithNeed])
        house.careSupplyFromWealth = 0
        if house.totalLifeExpectancy > 0:
            house.wealthForCare = sum([x.wealthForCare for x in peopleWithNeed])
            weeklyWealth = sum([x.wealthForCare for x in peopleWithNeed]) # /float(52*house.totalLifeExpectancy)
            house.careSupplyFromWealth = float(int(house.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])))
            
#            if n == 10:
#                print 'House life expectancy: ' + str(house.totalLifeExpectancy)
#                print 'Weekly wealth: ' + str(weeklyWealth)
#                print 'Care supply from wealth: ' + str(house.careSupplyFromWealth)
#                print 'Old wealth for care: ' + str(oldWealthForCare)
#                print 'New wealth for care: ' + str(house.wealthForCare)
                
        
        if supplier in house.suppliers:
        
            supplierIndex = house.suppliers.index(supplier)
            
            careFromWealth = 0
            if supplier == house:
                careFromWealth = house.careSupplyFromWealth
            weightedCareFromWealth = careFromWealth #*self.p['weightCareFromWealth']
        
            distance = 0
            if supplier != house:
                distance = house.careNetwork[house][supplier]['distance']
                
            household = list(supplier.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            householdInformalSupply = []
            for i in range(4):
                householdInformalSupply.append(sum([x.residualInformalSupplies[i] for x in householdCarers]))
            informalSupply = 0
            if supplier.town == town:
                informalSupply = float(sum([x.residualInformalSupplies[distance] for x in householdCarers]))
            house.networkInformalSupplies[supplierIndex] = informalSupply
            weightedInformalSupply = informalSupply
            
            # print 'Updating formal care supply...'
            maxFormalSocialCare = float(self.updateFormalSocialCareSupplies(supplier, distance))
            weightedMaxFormalSocialCare = maxFormalSocialCare #*self.p['weightCareFromIncome']
    
            house.networkFormalSocialCareSupplies[supplierIndex] = maxFormalSocialCare
            
            house.networkTotalSupplies[supplierIndex] = informalSupply+maxFormalSocialCare+careFromWealth
            house.weightedTotalSupplies[supplierIndex] = weightedInformalSupply+weightedMaxFormalSocialCare+weightedCareFromWealth
            
            house.networkSupply = sum(house.networkTotalSupplies)
            
            if house.networkSupply >= oldSupply and n != 0 and n != 10:
                print 'Error: social care supply did not decrease!'
                print 'Case: ' + str(n)
                print house.id
                print [x.id for x in house.suppliers]
                print supplierIndex
                print supplier.id
                print ''
                print oldSupply
                print house.networkSupply
                print ''
                print oldInformalSupplies
                print house.networkInformalSupplies
                print ''
                print oldFormalCareSupplies
                print house.networkFormalSocialCareSupplies
                print ''
                print oldCareFromWealth
                print house.careSupplyFromWealth
                print ''
                print oldTotalSupplies
                print house.networkTotalSupplies
                
                sys.exit()    
        
    def updateSocialCareNetworkSupply_Ind(self, receiver, supplier, n):
        
        town = receiver.house.town
        
        oldSupply = receiver.networkSupply
        oldInformalSupplies = list(receiver.networkInformalSupplies)
        oldFormalCareSupplies = list(receiver.networkFormalSocialCareSupplies)
        oldTotalSupplies = list(receiver.networkTotalSupplies)
        oldWealthForCare = receiver.wealthForCare
        oldCareFromWealth = receiver.careSupplyFromWealth
        
        receiver.careSupplyFromWealth = float(int(receiver.wealthForCare/self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])))
        careFromWealth = receiver.careSupplyFromWealth
        # weightedCareFromWealth = careFromWealth*self.p['weightCareFromWealth']
        receiver.networkSupply = receiver.careSupplyFromWealth
        
        houses = [x.house for x in receiver.suppliers]
        if supplier.house in houses:
            
            supplier = [x for x in receiver.suppliers if x.house.id == supplier.house.id][0]
            supplierIndex = receiver.suppliers.index(supplier)
            
            # suppliers = list(receiver.careNetwork.neighbors(receiver))
            
#            if supplier not in suppliers:
#                print 'Error: supplier not in network!'
#                print receiver.id
#                print [x.id for x in receiver.suppliers]
#                print [x.id for x in suppliers]
#                sys.exit()
            
            distance = receiver.careNetwork[receiver][supplier]['distance']
                
            household = list(supplier.house.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
            householdInformalSupply = []
            for i in range(4):
                householdInformalSupply.append(sum([x.residualInformalSupplies[i] for x in householdCarers]))
            informalSupply = 0
            if supplier.house.town == town:
                informalSupply = float(sum([x.residualInformalSupplies[distance] for x in householdCarers]))
            receiver.networkInformalSupplies[supplierIndex] = informalSupply
            weightedInformalSupply = informalSupply
            
            # print 'Updating formal care supply...'
            maxFormalSocialCare = float(self.updateFormalSocialCareSupplies_Ind(supplier, distance))
            weightedMaxFormalSocialCare = maxFormalSocialCare #*self.p['weightCareFromIncome']
    
            receiver.networkFormalSocialCareSupplies[supplierIndex] = maxFormalSocialCare
            
            informalFactor = math.pow(informalSupply, self.p['betaInformalCare'])
            formalFactor = math.pow(maxFormalSocialCare, self.p['betaFormalCare'])
            
            receiver.networkTotalSupplies[supplierIndex] = informalSupply+maxFormalSocialCare
            receiver.weightedTotalSupplies[supplierIndex] = informalFactor+formalFactor
            
        receiver.networkSupply += sum(receiver.networkTotalSupplies)
        
        # receiver.networkSupply += receiver.careSupplyFromWealth
         
        if receiver.networkSupply >= oldSupply and n != 0 and n != 10:
            print 'Error: social care supply did not decrease!'
            print 'Case: ' + str(n)
            print receiver.id
            print [x.id for x in receiver.suppliers]
            print supplierIndex
            print supplier.id
            print ''
            print oldSupply
            print receiver.networkSupply
            print ''
            print oldInformalSupplies
            print receiver.networkInformalSupplies
            print ''
            print oldFormalCareSupplies
            print receiver.networkFormalSocialCareSupplies
            print ''
            print oldCareFromWealth
            print receiver.careSupplyFromWealth
            print ''
            print oldTotalSupplies
            print receiver.networkTotalSupplies
            
            sys.exit() 
                
    def transferInformalChildCare(self, receiver, supplier):
        distance = 0
        if receiver != supplier:
            distance = receiver.careNetwork[receiver][supplier]['distance']
        
        householdCarers = [x for x in supplier.occupants if x.residualInformalSupplies[distance] > 0]
        # Sort the suppliers in order of decreasing informal supply
        totInformalSupply = 0
        for member in householdCarers:
            member.residualInformalSupply = member.residualInformalSupplies[distance]
            totInformalSupply += member.residualInformalSupply
        householdCarers.sort(key=operator.attrgetter("residualInformalSupply"), reverse=True)
        
        if totInformalSupply == 0:
            print 'Error: totInformalSupply is zero!'
            sys.exit()
        
        # Decrease the suppliers' informal supply by the quantum of care.
        
       
        residualCare = min(self.p['quantumCare'], receiver.totalChildCareNeed)
        if residualCare < 1.0:
            residualCare = 1.0
            
        careTransferred = 0
        for i in householdCarers:
            careForNeed = min(i.residualInformalSupply, residualCare)
            if careForNeed < 1.0:
                careForNeed = 1.0
            i.childWork += careForNeed
            if careForNeed > 0:
                careTransferred += careForNeed
                for j in range(4):
                    i.residualInformalSupplies[j] -= careForNeed
                    i.residualInformalSupplies[j] = float(max(int(i.residualInformalSupplies[j]), 0))
                residualCare -= careForNeed
                if residualCare <= 0:
                    break
                
        if  receiver != supplier:
            receiver.networkSupport += careTransferred
            
        if careTransferred == 0:
            print 'Informal care'
            print 'Residual care: ' + str(residualCare)
            print 'Error: care transferred is equal to zero!'
            sys.exit()
        
        
        
        # Decrease the child care needs of the receiver
        children = [x for x in receiver.occupants if x.age > 0 and x.age < self.p['ageTeenagers'] and x.unmetChildCareNeed > 0]
        # preChildCareNeed = list([x.unmetChildCareNeed for x in children])
        
        # Because of the one-to-many nature of child care, the informal care provided decreases the care need of all the household's children
        for child in children:
            child.informalChildCareReceived += min(careTransferred, child.unmetChildCareNeed)
            child.unmetChildCareNeed -= careTransferred
            child.unmetChildCareNeed = float(max(int(child.unmetChildCareNeed), 0))
        # postChildCareNeed = list([x.unmetChildCareNeed for x in children])
        receiver.totalChildCareNeed = sum([x.unmetChildCareNeed for x in children])
        receiver.informalChildCareReceived += careTransferred
        
#        preFormalSupply = self.updateFormalChildCareSupplies(receiver)
#        
#        preChildCareNeeds = receiver.childCareNeeds
#        preChildCarePrices = receiver.childCarePrices
#        preCumulatedChildren = receiver.cumulatedChildren
#        workers = [x for x in receiver.occupants if x.status == 'worker' and x.availableWorkingHours > 0]
#        workers.sort(key=operator.attrgetter("wage"))
#        availableHours = [x.availableWorkingHours for x in workers]
#        wages = [x.wage for x in workers]
#        formalChilCareReceived = [x.formalChildCareReceived for x in children]
#        discountedNeed = [max(self.p['childcareTaxFreeCap']-x, 0) for x in formalChilCareReceived]
        
        
        self.updateChildCareNeeds(receiver)
        
#        postFormalSupply = self.updateFormalChildCareSupplies(receiver)
#        
#        if postFormalSupply > preFormalSupply:
#            print 'Care Transferred: ' + str(careTransferred)
#            print ''
#            print preChildCareNeed
#            print postChildCareNeed
#            print '' 
#            print 'Pre Formal Supply: ' + str(preFormalSupply)
#            print 'Post formal supply: ' + str(postFormalSupply)
#            print ''
#            print preChildCareNeeds
#            print preChildCarePrices
#            print preCumulatedChildren
#            print ''
#            print receiver.childCareNeeds
#            print receiver.childCarePrices
#            print receiver.cumulatedChildren
#            print ''
#            print availableHours
#            print wages
#            print formalChilCareReceived
#            print discountedNeed
#            sys.exit()
        
        # print 'Checking netwrk supply after case 1'
        self.updateChildCareNetworkSupply(receiver, supplier, 1)
        
    def outOfIncomeChildCare(self, receiver):
        
#        print receiver.residualIncomeForChildCare
#        for x in receiver.occupants:
#            print x.status
#            print x.careNeedLevel
#            if x.status == 'worker':
#                print x.potentialIncome
#                print x.residualWorkingHours
#            print x.income
#        print 'Household income: ' + str(self.computeHouseholdIncome(receiver))
        
        employed = [x for x in receiver.occupants if x.status == 'worker' and x.availableWorkingHours > 0]
        children = [x for x in receiver.occupants if x.age > 0 and x.age < self.p['ageTeenagers'] and x.unmetChildCareNeed > 0]
        
        maxFormalSupply = receiver.formalChildCareSupply
        careSupply = min(self.p['quantumCare'], receiver.totalChildCareNeed, maxFormalSupply)
        
        if careSupply == 0:
            print 'Error: careSupply OOI is equal to zero!'
            print ''
            sys.exit()
        
        if careSupply < 1.0:
            careSupply = 1.0
        
        potentialCostChildCare = self.computePotentialFormalChildCareCost(receiver, careSupply)
        valueInformalChildCare = potentialCostChildCare/careSupply
        
        if len(employed) > 0:
            employed.sort(key=operator.attrgetter("wage"))
            carer = employed[0]
            
        if len(employed) == 0 or carer.wage >= valueInformalChildCare: # In this case, it is more convenient to pay formal care 
           
            costQuantumChildCare = self.childCareCost(children, careSupply)
            receiver.residualIncomeForChildCare -= costQuantumChildCare
            receiver.residualIncomeForChildCare = max(receiver.residualIncomeForChildCare, 0)
            receiver.householdFormalSupplyCost += costQuantumChildCare
            
            totalCareResources = receiver.residualIncomeForChildCare + receiver.householdFormalSupplyCost
            income = self.computeHouseholdIncome(receiver)
            
#            if int(totalCareResources) != int(income):
#                print receiver.householdFormalSupplyCost
#                print income
#                print receiver.residualIncomeForChildCare
#                print costQuantumChildCare
#                print 'Error: residual income greater than income (2)'
#                sys.exit()
            
#            print ''
#            print 'Pre formal child care: ' + str(self.updateFormalChildCareSupplies(receiver))
#            children = [x for x in receiver.occupants if x.age > 0 and x.age < self.p['ageTeenagers']]
#            print 'Max formal care: ' + str(sum([max(self.p['maxFormalChildCare'] - x.formalChildCareReceived, 0) for x in children]))
#            print [x.formalChildCareReceived for x in children]
            
            #Reducing the available working hours for informal care
            careTransferred = 0
            if len(employed) > 0:
                employed.sort(key=operator.attrgetter("wage"), reverse=True)
                residualCare = careSupply
                for worker in employed:
                    
                    maxCare = costQuantumChildCare/worker.wage
                    workerCare = min(maxCare, residualCare)
                    if workerCare < 1.0:
                        workerCare = 1.0
                    careTransferred += workerCare
                    
                    worker.incomeExpenses += workerCare*worker.wage
                    
                    worker.availableWorkingHours -= workerCare
                    worker.availableWorkingHours = float(max(int(worker.availableWorkingHours), 0))
                    costQuantumChildCare -= workerCare*worker.wage
                    residualCare -= workerCare
                    if residualCare <= 0:
                        break
            else:
                careTransferred = careSupply
                #employed.sort(key=operator.attrgetter("wage"), reverse=True)
                #employed[0].availableWorkingHours -= costQuantumChildCare/employed[0].wage
                #employed[0].availableWorkingHours = max(employed[0].availableWorkingHours, 0)
                
            formalChildCares = [x.formalChildCareReceived for x in children if x.formalChildCareReceived < self.p['childcareTaxFreeCap']]
            if len(formalChildCares) > 0:
                children.sort(key=operator.attrgetter("formalChildCareReceived"))
            else:
                children.sort(key=operator.attrgetter("unmetChildCareNeed"), reverse = True)
            
            if careTransferred == 0:
                print 'Out-of-income formal care'
                print 'Error: care transferred is equal to zero!'
                sys.exit()
            
            residualCare = careTransferred
            for child in children:
                careForChild = min(child.unmetChildCareNeed, residualCare)
                if careForChild > 0:
                    child.formalChildCareReceived += careForChild
                    child.unmetChildCareNeed -= careForChild
                    child.unmetChildCareNeed = float(max(int(child.unmetChildCareNeed), 0))
                residualCare -= careForChild
                if residualCare <= 0:
                    break
            receiver.totalChildCareNeed = sum([x.unmetChildCareNeed for x in children])
            receiver.formalChildCareReceived += careTransferred 
            receiver.formalChildCareCost += costQuantumChildCare
            
            
            self.updateChildCareNeeds(receiver)
            
#            print 'Post formal child care: ' + str(self.updateFormalChildCareSupplies(receiver))
#            children = [x for x in receiver.occupants if x.age > 0 and x.age < self.p['ageTeenagers']]
#            print 'Max formal care: ' + str(sum([max(self.p['maxFormalChildCare'] - x.formalChildCareReceived, 0) for x in children]))
#            print [x.formalChildCareReceived for x in children]
#            print ''
            # print 'Checking netwrk supply after case 2 (formal)'
            self.updateChildCareNetworkSupply(receiver, receiver, 2)
            
        else: # In this case, it is more convenient to take time off work to provide informal care
            
            
            # print 'Exectuing the time-off-work function'
            
            incomeBefore = self.computeHouseholdIncome(receiver)
            # print 'Household income before: ' + str(incomeBefore)
            
            residualIncomeBefore = receiver.residualIncomeForChildCare
            # print 'Residual Income for child care before: ' + str(residualIncomeBefore)
            
            childcareExpense = receiver.householdFormalSupplyCost
            # print 'Childcare expense: ' + str(childcareExpense)
            total = residualIncomeBefore+childcareExpense
            
#            if int(incomeBefore) != int(total):
#                print 'Error before'
#                sys.exit()
            
            # Care transferred
            careTransferred = min(careSupply, carer.availableWorkingHours)
            if careTransferred < 1.0:
                careTransferred = 1.0
            # print 'Care transferred: ' + str(careTransferred)
            if careTransferred == 0:
                print 'Out-of-income informal care'
                print 'Error: care transferred is equal to zero!'
                sys.exit()
            
            carer.availableWorkingHours -= careTransferred #self.p['quantumCare']
            carer.availableWorkingHours = float(max(0, int(carer.availableWorkingHours)))
            carer.residualWorkingHours -= careTransferred
            carer.residualWorkingHours = float(max(0, int(carer.residualWorkingHours)))
            carer.outOfWorkChildCare += careTransferred
            carer.childWork += careTransferred #self.p['quantumCare']
            receiver.residualIncomeForChildCare -= carer.wage*careTransferred #self.p['quantumCare']
            # print 'Miised income for care: ' + str(carer.wage*self.p['quantumCare'])
            
            
            receiver.residualIncomeForChildCare = max(receiver.residualIncomeForChildCare, 0)
           
            incomeAfter = self.computeHouseholdIncome(receiver)
            # print 'Household income After: ' + str(incomeAfter)
            
            residualIncomeAfter = receiver.residualIncomeForChildCare
            # print 'Residual Income for child care After: ' + str(residualIncomeAfter)
            
            total = residualIncomeAfter+childcareExpense
            
            totalCareResources = receiver.residualIncomeForChildCare + receiver.householdFormalSupplyCost
            income = self.computeHouseholdIncome(receiver)
            
#            if int(totalCareResources) != int(income):
#                print 'Error: residual income greater than income (3)'
#                print receiver.householdFormalSupplyCost
#                print income
#                print receiver.residualIncomeForChildCare
#                print incomeBefore
#                print incomeAfter
#                print total
#                print carer.wage*self.p['quantumCare']
#                print residualIncomeBefore
#                print residualIncomeAfter
#                for x in receiver.occupants:
#                    print x.status
#                    print x.careNeedLevel
#                    if x.status == 'worker':
#                        print x.potentialIncome
#                        print x.residualWorkingHours
#                    print x.income
                
                
                # sys.exit()
            
            # print incomeBefore - incomeAfter
            # print residualIncomeBefore - residualIncomeAfter
            
#            if int(incomeAfter) != int(total):
#                print incomeBefore
#                print incomeAfter
#                print total
#                print carer.wage*self.p['quantumCare']
#                print residualIncomeBefore
#                print residualIncomeAfter
#                print childcareExpense
#                
#                print 'Error after'
#                sys.exit()
            
            
            children = [x for x in receiver.occupants if x.age > 0 and x.age < self.p['ageTeenagers'] and x.unmetChildCareNeed > 0]
            for child in children:
                child.informalChildCareReceived += min(careTransferred, child.unmetChildCareNeed)
                child.unmetChildCareNeed -= careTransferred
                child.unmetChildCareNeed = float(max(int(child.unmetChildCareNeed), 0))
            receiver.totalChildCareNeed = sum([x.unmetChildCareNeed for x in children])
            receiver.informalChildCareReceived = sum([x.informalChildCareReceived for x in children])
            
            self.updateChildCareNeeds(receiver)
            
            
            # print 'Checking netwrk supply after case 2 (informal)'
            self.updateChildCareNetworkSupply(receiver, receiver, 3)
        
        
    def transferChildCare(self, receiver, supplier, index):
        case = -1
        # Transfer quantim of care: decide who trasfers which kind of care (informal or formal) to whom
        informalSupply = receiver.networkInformalSupplies[index]
        formalChildCare = 0
        if receiver == supplier:
            formalChildCare = receiver.formalChildCareSupply
        # Select kind of care based on supplier availability
        kindsOfCare = ['informal care', 'formal care']
        care = 'formal care'
        if supplier.town == receiver.town:
            careWeights = [informalSupply, formalChildCare]
            careProbs = [x/sum(careWeights) for x in careWeights]
            care = np.random.choice(kindsOfCare, p = careProbs) 
        
        # If 'informal care' is selected: informal care provider are sorted in decreasing order.
        # Their supply is used to satisfy the most expensive care need.
        if care == 'informal care':
            case = 3
#            print 'tranfer child care: informal (3)'
#            print informalSupply
            self.transferInformalChildCare(receiver, supplier)
        else:
            case = 4
#            print 'tranfer child care: formal (4)'
#            print formalChildCare
        # Both formal and out-of-income informal care are possible: choice depends on price of child care and lowest wage.
            self.outOfIncomeChildCare(receiver)
            
        return case
    
    def childCareCost(self, children, careSupply):
        children.sort(key=operator.attrgetter("formalChildCareReceived"))
        cost = 0
        residualCare = careSupply
        for child in children:
            careForChild = min(child.unmetChildCareNeed, residualCare)
            if careForChild + child.formalChildCareReceived <= self.p['childcareTaxFreeCap']:
                cost += self.p['priceChildCare']*(1.0-self.p['childCareTaxFreeRate'])*careForChild
                self.costTaxFreeChildCare += self.p['priceChildCare']*self.p['childCareTaxFreeRate']*careForChild
            else:
                if child.formalChildCareReceived >= self.p['childcareTaxFreeCap']:
                    cost += self.p['priceChildCare']*careForChild
                else:
                    discountedCare = self.p['childcareTaxFreeCap']-child.formalChildCareReceived
                    cost1 = discountedCare*self.p['priceChildCare']*(1.0-self.p['childCareTaxFreeRate'])
                    self.costTaxFreeChildCare += discountedCare*self.p['priceChildCare']*self.p['childCareTaxFreeRate']
                    fullPriceCare = careForChild - discountedCare
                    cost2 = fullPriceCare*self.p['priceChildCare']
                    cost += (cost1 + cost2)
            residualCare -= careForChild
            if residualCare <= 0:
                break
        return cost
    
    def computeHouseholdIncome(self, house):
        householdCarers = [x for x in house.occupants if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
        employed = [x for x in householdCarers if x.status == 'worker']
        householdIncome = 0
        for worker in employed:
            worker.income = worker.residualWorkingHours*worker.wage
            
        householdIncome = sum([x.income for x in householdCarers])

        return householdIncome
        
        
    def updateChildCareNeeds(self, house):
        children = [x for x in house.occupants if x.age > 0 and x.age < self.p['ageTeenagers']]
        children.sort(key=operator.attrgetter("unmetChildCareNeed"))
        residualNeeds = [x.unmetChildCareNeed for x in children]
        
        # print 'Unmet child care needs: ' + str(residualNeeds)
        
        marginalNeeds = []
        numbers = []
        toSubtract = 0
        for need in residualNeeds:
            marginalNeed = need-toSubtract
            marginalNeed = max(marginalNeed, 0)
            if marginalNeed > 0:
                marginalNeeds.append(marginalNeed)
                num = len([x for x in residualNeeds if x >= need])
                numbers.append(num)                
                toSubtract = need
        house.childCareNeeds = marginalNeeds
        house.cumulatedChildren = numbers
        
        # print 'House child care needs: ' + str(house.childCareNeeds)
        
        prices = []
        residualCare = 0
        cumulatedCare = 0
        for i in range(len(numbers)):
            cost = 0
            residualCare = house.childCareNeeds[i]
            for child in children[-numbers[i]:]:
                if cumulatedCare + residualCare + child.formalChildCareReceived <= self.p['childcareTaxFreeCap']:
                    cost += self.p['priceChildCare']*(1.0-self.p['childCareTaxFreeRate'])*residualCare
                else:
                    if child.formalChildCareReceived + cumulatedCare >= self.p['childcareTaxFreeCap']:
                        cost += self.p['priceChildCare']*residualCare
                    else:
                        discountedCare = self.p['childcareTaxFreeCap'] - (child.formalChildCareReceived + cumulatedCare)
                        cost1 = discountedCare*self.p['priceChildCare']*(1.0-self.p['childCareTaxFreeRate'])
                        fullPriceCare = residualCare - discountedCare
                        cost2 = fullPriceCare*self.p['priceChildCare']
                        cost += (cost1 + cost2)
            cumulatedCare += house.childCareNeeds[i]
            prices.append(cost/house.childCareNeeds[i])
        house.childCarePrices = prices
        
        # print 'House child care prices: ' + str(house.childCarePrices)
        
        house.highPriceChildCare = 0
        house.lowPriceChildCare = 0
        for i in range(len(house.childCarePrices)):
            if house.childCarePrices[i] >= self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate']):
                house.highPriceChildCare += house.childCareNeeds[i]
            else:
                house.lowPriceChildCare += house.childCareNeeds[i]
        
        
        if house.totalChildCareNeed > 0 and (house.highPriceChildCare+house.lowPriceChildCare) <= 0:
            print 'Error: mismatch between total child care needs in updateChildCareNeeds'
            print house.totalChildCareNeed
            print house.childCareNeeds
            print house.childCarePrices
            print house.highPriceChildCare
            print house.lowPriceChildCare
            sys.exit()
        # print 'High price care in updateChildCare: ' + str(house.highPriceChildCare)
            
        # print 'Low price care in updateChildCare: ' + str(house.lowPriceChildCare)
                    
    def computePotentialFormalChildCareCost(self, house, careSupply):
        # How much would the household have to pay to provide the same amount of care through formal childcare as a quantum of informal care?
        cost = 0
        residualNeed = careSupply
        for i in range(len(house.childCareNeeds)):
            careForChild = min(house.childCareNeeds[i], residualNeed)
            cost += careForChild*house.childCarePrices[i]
            residualNeed -= careForChild
            if residualNeed <= 0:
                break
        return cost
        
    def computeChildCareNetworkSupply(self, house):
        
        town = house.town
        
       
        
        house.networkSupply = 0
        house.formalChildCareSupply = 0
        house.networkTotalSupplies = []
        house.networkInformalSupplies = []
        house.childCareWeights = []
        house.formalCaresRatios = []
        house.suppliers = [house]
        house.suppliers.extend(list(house.careNetwork.neighbors(house)))
        
        # Household supply
        
#        household = list(house.occupants)
#        householdCarers = [x for x in household if x.residualInformalSupplies[0] > 0]
#        informalSupply = sum([x.residualInformalSupplies[0] for x in householdCarers])
#        house.networkInformalSupplies.append(informalSupply)
#        house.formalChildCareSupply = self.updateFormalChildCareSupplies(house)
#        householdTotalSupply = informalSupply + house.formalChildCareSupply
#        house.networkTotalSupplies.append(householdTotalSupply)
#        house.networkSupply = householdTotalSupply
#        
        for supplier in house.suppliers:
            
            if supplier == house:
                household = list(house.occupants)
                householdCarers = [x for x in household if x.residualInformalSupplies[0] > 0]
                informalSupply = sum([x.residualInformalSupplies[0] for x in householdCarers])
                house.networkInformalSupplies.append(informalSupply)
                house.formalChildCareSupply = self.updateFormalChildCareSupplies(house)
                householdTotalSupply = informalSupply + house.formalChildCareSupply
                house.networkTotalSupplies.append(householdTotalSupply)
                house.networkSupply += householdTotalSupply
            else:
                distance = house.careNetwork[house][supplier]['distance']
                household = list(supplier.occupants)
                householdCarers = [x for x in household if x.residualInformalSupplies[distance] > 0]
                informalSupply = 0
                if supplier.town == town:
                    informalSupply = sum([x.residualInformalSupplies[distance] for x in householdCarers])
                house.networkInformalSupplies.append(informalSupply)
                house.networkTotalSupplies.append(informalSupply)
                house.networkSupply += informalSupply
            
    def updateChildCareNetworkSupply(self, house, supplier, n):
        
        children = [x for x in house.occupants if x.age > 0 and x.age < self.p['ageTeenagers']]
        
        
        oldSupply = house.networkSupply
        oldFormalSupply = house.formalChildCareSupply
        oldInformalSupplies = list(house.networkInformalSupplies)
        
        town = house.town
        
        if supplier in house.suppliers:
        
            supplierIndex = house.suppliers.index(supplier)
        
            distance = 0
            if supplier != house:
                distance = house.careNetwork[house][supplier]['distance']
                
            household = list(supplier.occupants)
            householdCarers = [x for x in household if x.residualInformalSupplies[distance] > 0]
            informalSupply = 0
            if supplier.town == town:
                informalSupply = sum([x.residualInformalSupplies[distance] for x in householdCarers])
            house.networkInformalSupplies[supplierIndex] = informalSupply
            house.networkTotalSupplies[supplierIndex] = informalSupply
            
            if house == supplier:
                
               #  print 'Updating formal child care supply.....'
                residualFormalChilCare = sum([max(self.p['maxFormalChildCare'] - x.formalChildCareReceived, 0) for x in children])
                house.formalChildCareSupply = min(self.updateFormalChildCareSupplies(house), residualFormalChilCare)
                house.networkTotalSupplies[supplierIndex] += house.formalChildCareSupply

            house.networkSupply = sum(house.networkTotalSupplies)
            
#            if n != 4 and n != 3:
#                if house.networkSupply >= oldSupply and house:
#                    print ''
#                    print 'Case; ' + str(n)
#                    print oldSupply
#                    print house.networkSupply
#                    print ''
#                    print oldInformalSupplies
#                    print house.networkInformalSupplies
#                    print ''
#                    print oldFormalSupply
#                    print house.formalChildCareSupply
#                    print 'Error: child supply did not change'
#                    sys.exit()
            
    
    def updateIncomeByTaxBand(self, house):
        householdCarers = [x for x in house.occupants if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
        employed = [x for x in householdCarers if x.status == 'worker']
        householdIncome = 0
        for worker in employed:
            worker.income = worker.residualWorkingHours*worker.wage
        incomes = [x.income for x in householdCarers]
        
        taxBands = len(self.p['taxBrackets'])
        house.incomeByTaxBand = [0]*taxBands
        house.incomeByTaxBand[-1] = sum(incomes)
        for i in range(taxBands-1):
            for income in incomes:
                if income > self.p['taxBrackets'][i]:
                    bracket = income-self.p['taxBrackets'][i]
                    house.incomeByTaxBand[i] += bracket
                    house.incomeByTaxBand[-1] -= bracket
                    incomes[incomes.index(income)] -= bracket
        # Available income by tax band
        house.availableIncomeByTaxBand = house.incomeByTaxBand
        careExpense = house.householdFormalSupplyCost
        for i in range(len(house.availableIncomeByTaxBand)):
            if house.availableIncomeByTaxBand[i] > careExpense:
                house.availableIncomeByTaxBand[i] -= careExpense
                careExpense = 0
                break
            else:
                careExpense -= house.availableIncomeByTaxBand[i]
                house.availableIncomeByTaxBand[i] = 0
        return house.availableIncomeByTaxBand        
    
#    def updateFormalSocialCareSupplies(self, house, distance):
#        availableIncomeByTaxBand = list(self.updateIncomeByTaxBand(house))
#        residualIncomeForCare = house.residualIncomeForSocialCare[distance]
#        
#        # print 'residual Income for care: ' + str(residualIncomeForCare)
#        
#        # How much social care can the household buy with the residual income for care?
#        prices = [self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])*(1.0-x*self.p['taxBreakRate']) for x in self.p['taxationRates']]
#        incomeByTaxBand = list(availableIncomeByTaxBand)
#        totalHours = 0
#        for i in range(len(incomeByTaxBand)):
#            if residualIncomeForCare > incomeByTaxBand[i]:
#                totalHours += incomeByTaxBand[i]/prices[i]
#                residualIncomeForCare -= incomeByTaxBand[i]
#            else:
#                totalHours += residualIncomeForCare/prices[i]
#                break
#        
#        # print 'Tot hours: ' + str(totalHours)
#        
#        socialCareSupplies = round(totalHours)
#        
#        # socialCareSupplies = int((totalHours+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']
#        return socialCareSupplies
        
    def updateFormalSocialCareSupplies(self, house, distance):
        
        # The out-of-income social care is given by the sum of the informal social care provided by taking hours off work and the formal care paid through the
        # residual income.
        
        # availableIncomeByTaxBand = list(self.updateIncomeByTaxBand(house))
        residualIncomeForCare = house.residualIncomeForSocialCare[distance]
        workers = [x for x in house.occupants if x.status == 'worker' and x.availableWorkingHours > 0]
        workers.sort(key=operator.attrgetter("wage"))
        prices = [self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])*(1.0-x*self.p['taxBreakRate']) for x in self.p['taxationRates']]
        
        totHours = 0
        for worker in workers:
            if residualIncomeForCare == 0:
                break
            workerIncome = worker.availableWorkingHours*worker.wage
            for i in range(len(self.p['taxBrackets'])):
                bracket = max(workerIncome-self.p['taxBrackets'][i], 0)
                incomeForFormalCare = min(bracket, residualIncomeForCare)
                workerIncome -= incomeForFormalCare
                if prices[i] > worker.wage:
                    totHours += incomeForFormalCare/worker.wage
                else:
                    totHours += incomeForFormalCare/prices[i]
                residualIncomeForCare -= incomeForFormalCare
                residualIncomeForCare = max(residualIncomeForCare, 0)
                if residualIncomeForCare == 0:
                    break
        # print 'Tot hours: ' + str(totalHours)
        
        socialCareSupplies = int(totHours)
        
        # socialCareSupplies = int((totalHours+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']
        return socialCareSupplies    
    
    def updateFormalSocialCareSupplies_Ind(self, supplier, distance):
        
        # The out-of-income social care is given by the sum of the informal social care provided by taking hours off work and the formal care paid through the
        # residual income.
        house = supplier.house
        # availableIncomeByTaxBand = list(self.updateIncomeByTaxBand(house))
        residualIncomeForCare = house.residualIncomeForSocialCare[distance]
        workers = [x for x in house.occupants if x.status == 'worker' and x.availableWorkingHours > 0]
        workers.sort(key=operator.attrgetter("wage"))
        prices = [self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])*(1.0-x*self.p['taxBreakRate']) for x in self.p['taxationRates']]
        
        totHours = 0
        for worker in workers:
            if residualIncomeForCare == 0:
                break
            workerIncome = worker.availableWorkingHours*worker.wage
            for i in range(len(self.p['taxBrackets'])):
                bracket = max(workerIncome-self.p['taxBrackets'][i], 0)
                incomeForFormalCare = min(bracket, residualIncomeForCare)
                workerIncome -= incomeForFormalCare
                if prices[i] > worker.wage:
                    totHours += incomeForFormalCare/worker.wage
                else:
                    totHours += incomeForFormalCare/prices[i]
                residualIncomeForCare -= incomeForFormalCare
                residualIncomeForCare = max(residualIncomeForCare, 0)
                if residualIncomeForCare == 0:
                    break
        # print 'Tot hours: ' + str(totalHours)
        
        socialCareSupplies = int(totHours)
        
        # socialCareSupplies = int((totalHours+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']
        return socialCareSupplies

    def updateFormalChildCareSupplies(self, receiver):
        
        # residualIncomeForCare = receiver.residualIncomeForChildCare
        receiverOccupants = receiver.occupants
        children = [x for x in receiverOccupants if x.age > 0 and x.age < self.p['ageTeenagers']]
        workers = [x for x in receiverOccupants if x.status == 'worker' and x.availableWorkingHours > 0]
        workers.sort(key=operator.attrgetter("wage"))
        availableHours = [x.availableWorkingHours for x in workers]
        wages = [x.wage for x in workers]
        
#        print ''
#        print 'In-function children needs: ' + str(receiver.childCareNeeds)
#        print 'In-function child care prices: ' + str(receiver.childCarePrices)
#        print 'In-function num children: ' + str(receiver.cumulatedChildren)
#        print 'In-function available hours: ' + str(availableHours)
#        print 'In-function wages: ' + str(wages)
        
        
        residualNeeds = list(receiver.childCareNeeds)
        
        # print 'Wages: ' + str(wages)
        # print 'Available hours pre: ' + str(availableHours)
        
        totHours = 0
        for i in range(len(receiver.childCareNeeds)):
            for j in range(len(workers)):
                if receiver.childCarePrices[i] > wages[j]:
                    if availableHours[j] > residualNeeds[i]:
                        totHours += residualNeeds[i]*receiver.cumulatedChildren[i]
                        availableHours[j] -= residualNeeds[i]
                        residualNeeds[i] = 0
                        break
                    else:
                        totHours += availableHours[j]*receiver.cumulatedChildren[i]
                        residualNeeds[i] -= availableHours[j]
                        availableHours[j] = 0
                        
        # The remaining income is used to provide formal child care
        residualIncomeForCare = sum(np.multiply(availableHours,wages))
        
        # print 'Residual Incoem for Care: ' + str(residualIncomeForCare)
        # print 'Available hours post: ' + str(availableHours)
        # print 'Formal child care received:' + str(formalChilCareReceived)
        
        # List of residual formal child care below the tax free cap (i.e. for which the household get the discounted price)
        formalChilCareReceived = [x.formalChildCareReceived for x in children]
        
        # print 'Formal Child Care Received: ' + str(formalChilCareReceived)
        
        discountedNeed = [max(self.p['childcareTaxFreeCap']-x, 0) for x in formalChilCareReceived]
        
#        print 'In-function available hours: ' + str(formalChilCareReceived)
#        print 'In-function discounted need: ' + str(discountedNeed)
#        print ''
        
        # print 'Discounted Need: ' + str(discountedNeed)
        # print ''
        # print 'Discounted need: ' + str(discountedNeed)
        
        # Total cost of discounted child care 
        discountedCost = sum(discountedNeed)*self.p['priceChildCare']*(1.0-self.p['childCareTaxFreeRate'])
        
        
        
        # print 'Discounted Cost: ' + str(discountedCost)

        if residualIncomeForCare > discountedCost:
            totHours += sum(discountedNeed)
            residualIncomeForCare -= discountedCost
            totHours += residualIncomeForCare/self.p['priceChildCare']
        else:
            totHours += residualIncomeForCare/(self.p['priceChildCare']*(1.0-self.p['childCareTaxFreeRate']))
            
        # print 'Total formal supply hours: ' + str(totHours)
        
        # totHours = int((totHours+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']   
        totHours = int(totHours)
        
        return totHours

#    def updateFormalChildCareSupplies(self, receiver):
#        
#        residualIncomeForCare = receiver.residualIncomeForChildCare
#        receiverOccupants = receiver.occupants
#        children = [x for x in receiverOccupants if x.age > 0 and x.age < self.p['ageTeenagers']]
#        formalChilCareReceived = [x.formalChildCareReceived for x in children]
#        
#        # print 'Formal child care received:' + str(formalChilCareReceived)
#        
#        discountedNeed = [max(self.p['childcareTaxFreeCap']-x, 0) for x in formalChilCareReceived]
#        
#        # print 'Discounted need: ' + str(discountedNeed)
#        
#        discountedCost = sum(discountedNeed)*self.p['priceChildCare']*(1.0-self.p['childCareTaxFreeRate'])
#        
#        # print 'Discounted Cost: ' + str(discountedCost)
#        
#        totHours = 0
#        if residualIncomeForCare > discountedCost:
#            totHours = sum(discountedNeed)
#            residualIncomeForCare -= discountedCost
#            totHours += residualIncomeForCare/self.p['priceChildCare']
#        else:
#            totHours = residualIncomeForCare/(self.p['priceChildCare']*(1.0-self.p['childCareTaxFreeRate']))
#            
#        # print 'Total formal supply hours: ' + str(totHours)
#        
#        # totHours = int((totHours+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']   
#        totHours = round(totHours)
#        
#        return totHours
            
    def resetCareVariables_KN(self):
        
        for house in self.map.occupiedHouses:
            
            
            house.totalSocialCareNeed = 0
            house.totalUnmetSocialCareNeed = 0
            house.totalChildCareNeed = 0
            house.childCareNeeds = []
            house.childCarePrices = []
            house.highPriceChildCare = 0
            house.lowPriceChildCare = 0
            house.residualIncomeForChildCare = 0
            house.initialResidualIncomeForChildCare = 0
            house.residualIncomeForSocialCare = []
            house.householdInformalSupplies = []
            house.householdFormalSupply = []
            house.networkSupply = 0
            house.networkTotalSupplies = []
            house.networkInformalSupplies = []
            house.formalChildCareSupply = 0
            house.networkFormalSocialCareSupplies = []
            house.totalSupplies = []
            house.netCareSupply = 0
            house.childCareWeights = []
            house.formalCaresRatios = []
            house.informalChildCareReceived = 0
            house.informalSocialCareReceived = 0
            house.formalChildCareReceived = 0
            house.formalChildCareCost = 0
            house.formalSocialCareReceived = 0
            house.householdFormalSupplyCost = 0
            house.wealthForCare = 0
            house.incomeByTaxBand = []
            house.averageChildCarePrice = 0
            house.networkSupport = 0
            house.outOfWorkSocialCare = 0
            house.townAttractiveness = []
            house.netCareDemand = 0
            house.careAttractionFactor = 0
            house.newOccupancy = False
            
            for person in house.occupants:
                person.careNetwork.clear()
                person.hoursChildCareDemand = 0
                person.netChildCareDemand = 0
                person.unmetChildCareNeed = 0
                person.hoursSocialCareDemand = 0
                person.unmetSocialCareNeed = 0
                person.informalChildCareReceived = 0
                person.formalChildCareReceived = 0
                person.publicChildCareContribution = 0
                person.informalSocialCareReceived = 0
                person.formalSocialCareReceived = 0
                person.childWork = 0
                person.socialWork = 0
                person.potentialIncome = 0
                person.wealthPV = 0
                person.wealthForCare = 0
                person.incomeExpenses = 0
                person.outOfWorkChildCare = 0
                person.outOfWorkSocialCare = 0
                person.residualWorkingHours = 0
                person.availableWorkingHours = 0
                person.residualInformalSupplies = [0.0, 0.0, 0.0, 0.0]
                person.residualInformalSupply = 0
                person.hoursInformalSupplies = [0.0, 0.0, 0.0, 0.0]
                person.maxFormalCareSupply = 0
                person.totalSupply = 0
                person.informalSupplyByKinship = [0.0, 0.0, 0.0, 0.0]
                person.formalSupplyByKinship = [0.0, 0.0, 0.0, 0.0]
                person.careForFamily = False
                person.wealthSpentOnCare = 0
                # Social care provision variables
                person.networkSupply = 0
                person.networkTotalSupplies = []
                person.weightedTotalSupplies = []
                person.networkInformalSupplies = []
                person.networkFormalSocialCareSupplies = []
                person.suppliers = []
                person.networkInfectionFactor = 0
    
    def householdSocialCareNetwork(self):
        
        
        for house in self.map.occupiedHouses:
            
            house.careNetwork.clear()
            house.careNetwork.add_node(house, netDemand = 0)
            
            visited = []
            visited.append(house)
            
            household = list(house.occupants)
            residualCareNeed = max(sum([x.unmetSocialCareNeed for x in household])-house.totalSupplies[0], 0)
            # Distance 1
            for member in household:
                distanceOne = []
                if member.father != None:
                    nok = member.father
                    if nok.dead == False and nok not in household and nok.house not in visited:
                        distanceOne.append(nok.house)
                    nok = member.mother
                    if nok.dead == False and nok not in household and nok.house not in visited:
                        distanceOne.append(nok.house)
                for child in member.children:
                    nok = child
                    if nok.dead == False and nok not in household and nok.house not in visited:
                        distanceOne.append(nok.house)
            
            np.random.shuffle(distanceOne)
            
            for supplier in distanceOne:
                # Compute prob
                exponent = (residualCareNeed+1)/math.exp(self.p['distanceExp'])
                prob = (math.exp(self.p['networkExp']*exponent)-1)/math.exp(self.p['networkExp']*exponent)
                
                print 'Prob kinship network: ' + str(prob)
                
                if supplier not in visited and np.random.random() < prob:
                    house.careNetwork.add_node(supplier, netDemand = 0)
                    house.careNetwork.add_edge(house, supplier, distance = 1, careTransferred = 0)
                    residualCareNeed -= supplier.totalSupplies[1]
                    residualCareNeed = max(residualCareNeed, 0)
                    visited.append(supplier)
                
                
            # Distance 2
            for member in household:
                distanceTwo = []
                if member.father != None:
                    if member.father.father != None:
                        nok = member.father.father
                        if nok.dead == False and nok not in household and nok.house not in visited:
                            distanceTwo.append(nok.house)
                        nok = member.father.mother
                        if nok.dead == False and nok not in household and nok.house not in visited:
                            distanceTwo.append(nok.house)
                    if member.mother.father != None:
                        nok = member.mother.father
                        if nok.dead == False and nok not in household and nok.house not in visited:
                            distanceTwo.append(nok.house)
                        nok = member.mother.mother
                        if nok.dead == False and nok not in household and nok.house not in visited:
                            distanceTwo.append(nok.house)
                    brothers = list(set(member.father.children + member.mother.children))
                    brothers.remove(member)
                    for brother in brothers:
                        nok = brother
                        if nok.dead == False and nok not in household and nok.house not in visited:
                            distanceTwo.append(nok.house)
                for child in member.children:
                    for grandchild in child.children:
                        nok = grandchild
                        if nok.dead == False and nok not in household and nok.house not in visited:
                            distanceTwo.append(nok.house)
                            
            np.random.shuffle(distanceTwo)
            
            for supplier in distanceTwo:
                # Compute prob
                exponent = (residualCareNeed+1)/math.exp(self.p['distanceExp']*2.0)
                prob = (math.exp(self.p['networkExp']*exponent)-1)/math.exp(self.p['networkExp']*exponent)
                if supplier not in visited and np.random.random() < prob:
                    house.careNetwork.add_edge(house, supplier, distance = 2)
                    residualCareNeed -= supplier.totalSupplies[2]
                    residualCareNeed = max(residualCareNeed, 0)
                    visited.append(supplier)
                            
            # Distance 3
            for member in household:
                distanceThree = []
                uncles = []
                if member.father != None:
                    if member.father.father != None:
                        uncles = list(set(member.father.father.children + member.father.mother.children))
                        uncles.remove(member.father)
                    if member.mother.father != None:
                        uncles.extend(list(set(member.mother.father.children + member.mother.mother.children)))
                        uncles.remove(member.mother)
                    for uncle in uncles:
                        nok = uncle
                        if nok.dead == False and nok not in household and nok.house not in visited:
                            distanceThree.append(nok.house)
                    brothers = list(set(member.father.children + member.mother.children))
                    brothers.remove(member)
                    for brother in brothers:
                        for child in brother.children:
                            nok = child
                            if nok.dead == False and nok not in household and nok.house not in visited:
                                distanceThree.append(nok.house)
                                
            np.random.shuffle(distanceThree)
            
            for supplier in distanceThree:
                # Compute prob
                exponent = (residualCareNeed+1)/math.exp(self.p['distanceExp']*3.0)
                prob = (math.exp(self.p['networkExp']*exponent)-1)/math.exp(self.p['networkExp']*exponent)
                if supplier not in visited and np.random.random() < prob:
                    house.careNetwork.add_edge(house, supplier, distance = 3)
                    residualCareNeed -= supplier.totalSupplies[3]
                    residualCareNeed = max(residualCareNeed, 0)
                    visited.append(supplier)
                    
                    
    def householdCareNetwork_netSupply(self):
        
        for house in self.map.occupiedHouses:
            house.careNetwork.clear()
            house.careNetwork.add_node(house)
            
            visited = []
            visited.append(house)
            
            household = list(house.occupants)
            residualCareNeed = sum([x.unmetSocialCareNeed for x in household])
            # Distance 1
            for member in household:
                
                if member.father != None:
                    nok = member.father
                    if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                        house.careNetwork.add_edge(house, nok.house, distance = 1)
                        visited.append(nok.house)
                    nok = member.mother
                    if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                        house.careNetwork.add_edge(house, nok.house, distance = 1)
                        visited.append(nok.house)
                for child in member.children:
                    nok = child
                    if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                        house.careNetwork.add_edge(house, nok.house, distance = 1)
                        visited.append(nok.house)
                        
            # Distance 2
            for member in household:
                if member.father != None:
                    if member.father.father != None:
                        nok = member.father.father
                        if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                            house.careNetwork.add_edge(house, nok.house, distance = 2)
                            visited.append(nok.house)
                        nok = member.father.mother
                        if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                            house.careNetwork.add_edge(house, nok.house, distance = 2)
                            visited.append(nok.house)
                    if member.mother.father != None:
                        nok = member.mother.father
                        if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                            house.careNetwork.add_edge(house, nok.house, distance = 2)
                            visited.append(nok.house)
                        nok = member.mother.mother
                        if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                            house.careNetwork.add_edge(house, nok.house, distance = 2)
                            visited.append(nok.house)
                    brothers = list(set(member.father.children + member.mother.children))
                    brothers.remove(member)
                    for brother in brothers:
                        nok = brother
                        if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                            house.careNetwork.add_edge(house, nok.house, distance = 2)
                            visited.append(nok.house)
                for child in member.children:
                    for grandchild in child.children:
                        nok = grandchild
                        if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                            house.careNetwork.add_edge(house, nok.house, distance = 2)
                            visited.append(nok.house)
                            
            # Distance 3
            for member in household:
                uncles = []
                if member.father != None:
                    if member.father.father != None:
                        uncles = list(set(member.father.father.children + member.father.mother.children))
                        uncles.remove(member.father)
                    if member.mother.father != None:
                        uncles.extend(list(set(member.mother.father.children + member.mother.mother.children)))
                        uncles.remove(member.mother)
                    for uncle in uncles:
                        nok = uncle
                        if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                            house.careNetwork.add_edge(house, nok.house, distance = 3)
                            visited.append(nok.house)
                    brothers = list(set(member.father.children + member.mother.children))
                    brothers.remove(member)
                    for brother in brothers:
                        for child in brother.children:
                            nok = child
                            if nok.dead == False and nok not in household and nok.house not in visited and nok.house.netCareSupply > 0:
                                house.careNetwork.add_edge(house, nok.house, distance = 3)
                                visited.append(nok.house)
                                      
    def computeSocialCareNeeds(self):
        
        self.publicSocialCare = 0
        for house in self.map.occupiedHouses:
            household = list(house.occupants)
            for person in household:
                careNeed = self.p['careDemandInHours'][person.careNeedLevel]
                person.hoursSocialCareDemand = careNeed
                person.unmetSocialCareNeed = person.hoursSocialCareDemand
                
                if person.unmetSocialCareNeed > 0:
                    person.wealthPV = person.financialWealth*math.pow(1.0 + self.p['pensionReturnRate'], person.lifeExpectancy)
                    person.wealthForCare = person.financialWealth
                
                preCareNeed = person.unmetSocialCareNeed
                
                if person.careNeedLevel >= self.p['publicCareNeedLevel'] and person.age >= self.p['publicCareAgeLimit'] and person.independentStatus == True:
                    socialCareCost = person.unmetSocialCareNeed*self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])
                    # The state pays for all the social care need that cannot be satisfied by the person with his income (leaving him a minimum income)
                    if socialCareCost > person.income - self.p['minimumIncomeGuarantee']:
                        if person.income > self.p['minimumIncomeGuarantee']:
                            stateContribution = 0
                            if person.wealth <= self.p['minWealthMeansTest']:
                                stateContribution = (socialCareCost - (person.income - self.p['minimumIncomeGuarantee']))
                            elif  person.wealth > self.p['minWealthMeansTest'] and person.wealth < self.p['maxWealthMeansTest']:
                                stateContribution = (socialCareCost - (person.income - self.p['minimumIncomeGuarantee']))
                                stateContribution -= int(person.wealth/self.p['wealthToPoundReduction'])
                                stateContribution = max(stateContribution, 0)
                            stateCare = stateContribution/(self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate']))
                            stateCare = int((stateCare+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']
                            # print 'State Care: ' + str(stateCare)
                            if stateCare < 0:
                                print 'Error: public care is negative!'
                                sys.exit()
                            self.publicSocialCare += stateCare
                            person.unmetSocialCareNeed -= stateCare   
                        else:
                            stateContribution = 0
                            if person.wealth <= self.p['minWealthMeansTest']:
                                stateContribution = socialCareCost
                            elif  person.wealth > self.p['minWealthMeansTest'] and person.wealth < self.p['maxWealthMeansTest']:
                                stateContribution = socialCareCost
                                stateContribution -= int(person.wealth/self.p['wealthToPoundReduction'])
                                stateContribution = max(stateContribution, 0)
                            stateCare = stateContribution/(self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate']))
                            stateCare = int((stateCare+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']
                            # print 'State Care: ' + str(stateCare)
                            if stateCare < 0:
                                print 'Error: public care is negative!'
                                sys.exit()
                            self.publicSocialCare += stateCare
                            person.unmetSocialCareNeed -= stateCare
                        
                        postCareNeed = person.unmetSocialCareNeed
                        
                        if postCareNeed > preCareNeed or postCareNeed < 0:
                            print postCareNeed
                            print preCareNeed
                            print person.income
                            print stateContribution
                            print stateCare
                            # sys.exit()
                            
                        
            house.totalSocialCareNeed = sum([x.hoursSocialCareDemand for x in household])
            house.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in household])
        
        self.costPublicSocialCare = self.publicSocialCare*self.p['priceSocialCare']
        self.publicCareProvision.append(self.publicSocialCare)
        totalSocialCareNeed = sum([x.hoursSocialCareDemand for x in self.pop.livingPeople if x.careNeedLevel > 0])
        totalResidualSocialCareNeed = sum([x.unmetSocialCareNeed for x in self.pop.livingPeople if x.careNeedLevel > 0])
        self.sharePublicSocialCare = 0
        if totalSocialCareNeed > 0:
            self.sharePublicSocialCare = 1.0 - float(totalResidualSocialCareNeed)/float(totalSocialCareNeed)
            
            
    def computeSocialCareNeeds_W(self):
        
        self.publicSocialCare = 0
        for house in self.map.occupiedHouses:
            household = list(house.occupants)
            for person in household:
                careNeed = self.p['careDemandInHours'][person.careNeedLevel]
                # Adjust care needs to account for virus symptoms
                if person.symptomatic == True:
                    if person.symptomsLevel == 'severe' or person.symptomsLevel == 'critical' or person.symptomsLevel == 'dead':
                        careNeed = 0 # The person is hospitalized, so the hospital takes care of all the care needs
                    elif person.symptomsLevel == 'mild':
                        # People not hospitalized, depending on the severity of their condition, develop care needs.
                        covidCareFactor = person.mildConditionIndex - self.p['symptomSocialCareThreshold']
                        if covidCareFactor > 0:
                            careNeed += float(self.p['careDemandInHours'][4]-careNeed)*covidCareFactor
                            careNeed = int((careNeed+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']
                    
                person.hoursSocialCareDemand = careNeed
                person.unmetSocialCareNeed = person.hoursSocialCareDemand
                
                person.wealthForCare = 0
                if person.unmetSocialCareNeed > 0:
                    person.wealthPV = person.financialWealth*math.pow(1.0 + self.p['pensionReturnRate'], person.lifeExpectancy)
                    
                    shareWealthForCare = 1.0 - 1.0/math.exp(self.p['wealthCareParam']*person.financialWealth)
                    
                    # print 'Share of financial wealth for care: ' + str(shareWealthForCare)
                    
                    if shareWealthForCare < 0:
                        print 'Error: negative share: ' + str(person.financialWealth)
                        sys.exit()
                    
                    person.wealthForCare = ((person.financialWealth/person.lifeExpectancy)*shareWealthForCare)/365.0
                
                preCareNeed = person.unmetSocialCareNeed
                
                # Compute probability of taking advantage of public social care
                # Depends on income (-) of household and level of unmet care need (+)
                
                if person.careNeedLevel >= self.p['publicCareNeedLevel'] and person.age >= self.p['publicCareAgeLimit'] and person.independentStatus == True:
                    socialCareCost = person.unmetSocialCareNeed*self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate'])
                    # The state pays for all the social care need that cannot be satisfied by the person with his income (leaving him a minimum income)
                    if socialCareCost > person.income - self.p['minimumIncomeGuarantee']:
                        if person.income > self.p['minimumIncomeGuarantee']:
                            stateContribution = 0
                            if person.financialWealth <= self.p['minWealthMeansTest']:
                                stateContribution = (socialCareCost - (person.income - self.p['minimumIncomeGuarantee']))
                            elif  person.financialWealth > self.p['minWealthMeansTest'] and person.financialWealth < self.p['maxWealthMeansTest']:
                                stateContribution = (socialCareCost - (person.income - self.p['minimumIncomeGuarantee']))
                                stateContribution -= int(person.financialWealth/self.p['wealthToPoundReduction'])
                                stateContribution = max(stateContribution, 0)
                            stateCare = stateContribution/(self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate']))
                            stateCare = int((stateCare+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']
                            # print 'State Care: ' + str(stateCare)
                            if stateCare < 0:
                                print 'Error: public care is negative!'
                                sys.exit()
                            self.publicSocialCare += stateCare
                            person.unmetSocialCareNeed -= stateCare   
                        else:
                            stateContribution = 0
                            if person.financialWealth <= self.p['minWealthMeansTest']:
                                stateContribution = socialCareCost
                            elif  person.financialWealth > self.p['minWealthMeansTest'] and person.financialWealth < self.p['maxWealthMeansTest']:
                                stateContribution = socialCareCost
                                stateContribution -= int(person.financialWealth/self.p['wealthToPoundReduction'])
                                stateContribution = max(stateContribution, 0)
                            stateCare = stateContribution/(self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate']))
                            stateCare = int((stateCare+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare']
                            # print 'State Care: ' + str(stateCare)
                            if stateCare < 0:
                                print 'Error: public care is negative!'
                                sys.exit()
                            self.publicSocialCare += stateCare
                            person.unmetSocialCareNeed -= stateCare
                        
                        postCareNeed = person.unmetSocialCareNeed
                        
                        if postCareNeed > preCareNeed or postCareNeed < 0:
                            print postCareNeed
                            print preCareNeed
                            print person.income
                            print stateContribution
                            print stateCare
                            # sys.exit()
                            
                        
            house.totalSocialCareNeed = sum([x.hoursSocialCareDemand for x in household])
            house.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in household])
        
        self.costPublicSocialCare = self.publicSocialCare*self.p['priceSocialCare']
        self.publicCareProvision.append(self.publicSocialCare)
        totalSocialCareNeed = sum([x.hoursSocialCareDemand for x in self.pop.livingPeople if x.careNeedLevel > 0])
        totalResidualSocialCareNeed = sum([x.unmetSocialCareNeed for x in self.pop.livingPeople if x.careNeedLevel > 0])
        self.sharePublicSocialCare = 0
        if totalSocialCareNeed > 0:
            self.sharePublicSocialCare = 1.0 - float(totalResidualSocialCareNeed)/float(totalSocialCareNeed)
       
    def computeChildCareNeeds(self):
        
        self.publicChildCare = 0
        for house in self.map.occupiedHouses:
        
            household = list(house.occupants)
            
            children = [x for x in household if x.age > 0 and x.age < self.p['ageTeenagers']]
            
            newBorns = [x for x in household if x.age == 0]
            for child in newBorns:
                child.hoursChildCareDemand = self.p['zeroYearCare']
                child.netChildCareDemand = child.hoursChildCareDemand
                child.informalChildCareReceived = self.p['zeroYearCare']
                child.unmetChildCareNeed = 0
                child.mother.childWork = self.p['zeroYearCare']
            
            for child in children:
                child.hoursChildCareDemand = max(0, self.p['childCareDemand'] - child.unmetSocialCareNeed)
            
#            householdCarers = [x for x in house.occupants if x.age >= self.p['ageTeenagers'] and x.maternityStatus == False]
#            employed = [x for x in householdCarers if x.status == 'worker']
#            householdIncome = 0
#            for worker in employed:
#                worker.income = worker.residualWorkingHours*worker.wage
                    
            childrenInHospital = [x for x in children if x.symptomatic == True and (x.symptomsLevel == 'severe' or x.symptomsLevel == 'critical' or x.symptomsLevel == 'dead')]  
            for child in childrenInHospital:
                child.unmetChildCareNeed = 0
                child.netChildCareDemand = child.unmetChildCareNeed
            
            children = [x for x in children if x not in childrenInHospital]
                
                
            house.totalChildCareNeed = 0
            income = sum([x.income for x in household])
            if income  < self.p['maxHouseholdIncomeChildCareSupport']:
                firstGroup = [x for x in children if x.age < 2]
                for child in firstGroup:
                    child.unmetChildCareNeed = child.hoursChildCareDemand
                    child.netChildCareDemand = child.unmetChildCareNeed
                    house.totalChildCareNeed += child.unmetChildCareNeed
                secondGroup = [x for x in children if x.age  == 2]
                for child in secondGroup:
                    hoursInSchool = self.p['freeChildCareHoursToddlers']
                    if child.symptomatic == True:
                        if self.lockdown == True or (child.symptomsLevel == 'mild' and child.mildConditionIndex > self.p['symptomChildcareThreshold']):
                            hoursInSchool = 0
                    child.unmetChildCareNeed = max(0, child.hoursChildCareDemand - hoursInSchool)
                    child.netChildCareDemand = child.unmetChildCareNeed
                    house.totalChildCareNeed += child.unmetChildCareNeed
                    self.publicChildCare += min(child.hoursChildCareDemand, hoursInSchool)
                thirdGroup = [x for x in children if x.age > 2 and x.age < 5]
                for child in thirdGroup:
                    hoursInSchool = self.p['freeChildCareHoursPreSchool']
                    if child.symptomatic == True:
                        if self.lockdown == True or (child.symptomsLevel == 'mild' and child.mildConditionIndex > self.p['symptomChildcareThreshold']):
                            hoursInSchool = 0
                    child.unmetChildCareNeed = max(0, child.hoursChildCareDemand - hoursInSchool)
                    child.netChildCareDemand = child.unmetChildCareNeed
                    house.totalChildCareNeed += child.unmetChildCareNeed
                    self.publicChildCare += min(child.hoursChildCareDemand, hoursInSchool)
                fourthGroup = [x for x in children if x.age > 4]
                for child in fourthGroup:
                    hoursInSchool = self.p['symptomChildcareThreshold']
                    if child.symptomatic == True:
                        if self.lockdown == True or (child.symptomsLevel == 'mild' and child.mildConditionIndex > self.p['symptomChildcareThreshold']):
                            hoursInSchool = 0
                    child.unmetChildCareNeed = max(0, child.hoursChildCareDemand - hoursInSchool)
                    child.netChildCareDemand = child.unmetChildCareNeed
                    house.totalChildCareNeed += child.unmetChildCareNeed
                    self.publicChildCare += min(child.hoursChildCareDemand, hoursInSchool)
            else:
                firstGroup = [x for x in children if x.age < 3]
                for child in firstGroup:
                    child.unmetChildCareNeed = child.hoursChildCareDemand
                    child.netChildCareDemand = child.unmetChildCareNeed
                    house.totalChildCareNeed += child.unmetChildCareNeed
                secondGroup = [x for x in children if x.age > 2 and x.age < 5]
                for child in secondGroup:
                    hoursInSchool = self.p['freeChildCareHoursPreSchool']
                    if child.symptomatic == True:
                        if self.lockdown == True or (child.symptomsLevel == 'mild' and child.mildConditionIndex > self.p['symptomChildcareThreshold']):
                            hoursInSchool = 0
                    child.unmetChildCareNeed = max(0, child.hoursChildCareDemand - hoursInSchool)
                    child.netChildCareDemand = child.unmetChildCareNeed
                    house.totalChildCareNeed += child.unmetChildCareNeed
                    self.publicChildCare += min(child.hoursChildCareDemand, hoursInSchool)
                thirdGroup = [x for x in children if x.age > 4]
                for child in thirdGroup:
                    hoursInSchool = self.p['freeChildCareHoursSchool']
                    if child.symptomatic == True:
                        if self.lockdown == True or (child.symptomsLevel == 'mild' and child.mildConditionIndex > self.p['symptomChildcareThreshold']):
                            hoursInSchool = 0
                    child.unmetChildCareNeed = max(0, child.hoursChildCareDemand - hoursInSchool)
                    child.netChildCareDemand = child.unmetChildCareNeed
                    house.totalChildCareNeed += child.unmetChildCareNeed
                    self.publicChildCare += min(child.hoursChildCareDemand, hoursInSchool)
                    
            children.sort(key=operator.attrgetter("unmetChildCareNeed"))
            
            totalNeed = sum([x.unmetChildCareNeed for x in children])
            discountedNeed = sum([min(self.p['childcareTaxFreeCap'], x.unmetChildCareNeed) for x in children])
            if totalNeed > 0:
                ratio = discountedNeed/totalNeed
            else:
                ratio = 0.0
            house.averageChildCarePrice = self.p['priceChildCare']*(1.0-ratio) + self.p['priceChildCare']*(1-self.p['childCareTaxFreeRate'])*ratio
                
            residualNeeds = [x.unmetChildCareNeed for x in children]
            marginalNeeds = []
            numbers = []
            toSubtract = 0
            for need in residualNeeds:
                marginalNeed = need-toSubtract
                if marginalNeed > 0:
                    marginalNeeds.append(marginalNeed)
                    num = len([x for x in residualNeeds if x >= need])
                    numbers.append(num)                
                    toSubtract = need
            house.childCareNeeds = marginalNeeds
            house.cumulatedChildren = numbers
            
            prices = []
            residualCare = 0
            cumulatedCare = 0
            for i in range(len(numbers)):
                cost = 0
                residualCare = house.childCareNeeds[i]
                for child in children[-numbers[i]:]:
                    if cumulatedCare + residualCare + child.formalChildCareReceived <= self.p['childcareTaxFreeCap']:
                        cost += self.p['priceChildCare']*(1-self.p['childCareTaxFreeRate'])*residualCare
                    else:
                        if child.formalChildCareReceived + cumulatedCare >= self.p['childcareTaxFreeCap']:
                            cost += self.p['priceChildCare']*residualCare
                        else:
                            discountedCare = self.p['childcareTaxFreeCap'] - (child.formalChildCareReceived + cumulatedCare)
                            cost1 = discountedCare*self.p['priceChildCare']*(1-self.p['childCareTaxFreeRate'])
                            fullPriceCare = residualCare - discountedCare
                            cost2 = fullPriceCare*self.p['priceChildCare']
                            cost += (cost1 + cost2)
                cumulatedCare += house.childCareNeeds[i]
                prices.append(cost/house.childCareNeeds[i])
            house.childCarePrices = prices
            
            # Compite high and low price chil care need
            house.highPriceChildCare = 0
            house.lowPriceChildCare = 0
            for i in range(len(house.childCarePrices)):
                if house.childCarePrices[i] >= self.p['priceSocialCare']*(1.0 - self.p['socialCareTaxFreeRate']):
                    house.highPriceChildCare += house.childCareNeeds[i]
                else:
                    house.lowPriceChildCare += house.childCareNeeds[i]
                    
            if house.totalChildCareNeed > 0 and (house.highPriceChildCare+house.lowPriceChildCare) <= 0:
                print 'Error: mismatch between total child care needs in computeChildCareNeeds'
                print residualNeeds
                print house.totalChildCareNeed
                print house.childCareNeeds
                print house.childCarePrices
                print house.highPriceChildCare
                print house.lowPriceChildCare
                sys.exit()
        
        children = [x for x in self.pop.livingPeople if x.age > 0 and x.age < self.p['ageTeenagers']]
        totalChildCareNeed = sum([x.hoursChildCareDemand for x in children])
        totalUnmetChildCareNeed = sum([x.unmetChildCareNeed for x in children])
      
        self.costPublicChildCare = self.publicChildCare*self.p['priceChildCare']
        self.sharePublicChildCare = 0
        if totalChildCareNeed > 0:
            self.sharePublicChildCare = 1.0 - float(totalUnmetChildCareNeed)/float(totalChildCareNeed)
            
    def householdCareSupply(self):
        
        for house in self.map.occupiedHouses:
        
            household = list(house.occupants)
            householdCarers = [x for x in household if x.age >= self.p['ageTeenagers'] and x.hoursSocialCareDemand == 0 and x.maternityStatus == False]
            for member in householdCarers:
                # Teenager supplies
                if member.status == 'teenager':
                    supplies = list(self.p['teenagerSupply'])
                    if member.symptomatic == True:
                        if member.symptomsLevel == 'severe' or member.symptomsLevel == 'critical' or member.symptomsLevel == 'dead' or member.mildConditionIndex > self.p['symptomSocialCareThreshold'] or member.testPositive == True:
                            supplies = [0 for x in supplies]
                        elif member.symptomsLevel == 'mild':
                            supplies = [float(x)*(1.0-member.mildConditionIndex) for x in supplies]
                    if self.lockdown == True:
                        homeSupply = supplies[0]
                        if member.symptomatic == False or member.mildConditionIndex < self.p['symptomSocialCareThreshold']:
                            homeSupply = float(supplies[0])*(1.0+self.p['increasedSupplyFactor'])
                        if self.p['careLockdown'] == False:
                            otherSupplies = [float(x)*(1.0+self.p['increasedSupplyFactor']) for x in supplies[1:]]
                        else:
                            otherSupplies = [0 for x in supplies[1:]]
                        supplies = [homeSupply]
                        supplies.extend(otherSupplies)
                    supplies = [int((x+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare'] for x in supplies]
                    member.residualInformalSupplies = supplies
                
                # Student supplies
                elif member.status == 'student' and member.outOfTownStudent == False:
                    supplies = list(self.p['studentSupply'])
                    if member.symptomatic == True:
                        if member.symptomsLevel == 'severe' or member.symptomsLevel == 'critical' or member.symptomsLevel == 'dead' or member.mildConditionIndex > self.p['symptomSocialCareThreshold'] or member.testPositive == True:
                            supplies = [0 for x in supplies]
                        elif member.symptomsLevel == 'mild':
                            supplies = [float(x)*(1.0-member.mildConditionIndex) for x in supplies]
                    if self.lockdown == True:
                        homeSupply = supplies[0]
                        if member.symptomatic == False or member.mildConditionIndex < self.p['symptomSocialCareThreshold']:
                            homeSupply = float(supplies[0])*(1.0+self.p['increasedSupplyFactor'])
                        if self.p['careLockdown'] == False:
                            otherSupplies = [float(x)*(1.0+self.p['increasedSupplyFactor']) for x in supplies[1:]]
                        else:
                            otherSupplies = [0 for x in supplies[1:]]
                        supplies = [homeSupply]
                        supplies.extend(otherSupplies)
                    supplies = [int((x+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare'] for x in supplies]
                    member.residualInformalSupplies = supplies
                
                # Retired supplies
                elif member.status == 'retired':
                    supplies = list(self.p['retiredSupply'])
                    if member.symptomatic == True:
                        if member.symptomsLevel == 'severe' or member.symptomsLevel == 'critical' or member.symptomsLevel == 'dead' or member.mildConditionIndex > self.p['symptomSocialCareThreshold'] or member.testPositive == True:
                            supplies = [0 for x in supplies]
                        elif member.symptomsLevel == 'mild':
                            supplies = [float(x)*(1.0-member.mildConditionIndex) for x in supplies]
                    if self.lockdown == True:
                        homeSupply = supplies[0]
                        if self.p['careLockdown'] == False:
                            otherSupplies = [float(x) for x in supplies[1:]]
                        else:
                            otherSupplies = [0 for x in supplies[1:]]
                        supplies = [homeSupply]
                        supplies.extend(otherSupplies)
                    supplies = [int((x+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare'] for x in supplies]
                    member.residualInformalSupplies = supplies
        
                # Workers supplies
                elif member.status == 'worker' and member.careNeedLevel == 0:
                    supplies = list(self.p['employedSupply'])
                    if member.symptomatic == True:
                        if member.symptomsLevel == 'severe' or member.symptomsLevel == 'critical' or member.symptomsLevel == 'dead' or member.mildConditionIndex > self.p['symptomSocialCareThreshold'] or member.testPositive == True:
                            supplies = [0 for x in supplies]
                        elif member.symptomsLevel == 'mild':
                            supplies = [float(x)*(1.0-member.mildConditionIndex) for x in supplies]
                    if self.lockdown == True:
                        homeSupply = supplies[0]
                        if member.symptomatic == False or member.mildConditionIndex < self.p['symptomSocialCareThreshold']:
                            homeSupply = float(supplies[0])*(1.0+self.p['increasedSupplyFactor'])
                        if self.p['careLockdown'] == False:
                            otherSupplies = [float(x)*(1.0+self.p['increasedSupplyFactor']) for x in supplies[1:]]
                        else:
                            otherSupplies = [0 for x in supplies[1:]]
                        supplies = [homeSupply]
                        supplies.extend(otherSupplies)
                    supplies = [int((x+self.p['quantumCare']/2)/self.p['quantumCare'])*self.p['quantumCare'] for x in supplies]
                    member.residualInformalSupplies = supplies
                    
                # member.residualInformalSupplies = [max(x-member.hoursSocialCareDemand, 0) for x in member.residualInformalSupplies]
                member.hoursInformalSupplies = member.residualInformalSupplies
                
            house.householdInformalSupplies = []
            for i in range(4):
                house.householdInformalSupplies.append(sum([x.residualInformalSupplies[i] for x in householdCarers]))
            
            employed = [x for x in householdCarers if x.status == 'worker']
            for worker in employed:
                if worker.careNeedLevel < 2:
                    worker.residualWorkingHours = self.p['dailyHours'][worker.careNeedLevel]
                    worker.availableWorkingHours = worker.residualWorkingHours
                worker.potentialIncome = self.p['dailyHours'][worker.careNeedLevel]*worker.wage
                
            potentialIncomes = [x.potentialIncome for x in household if x.maternityStatus == False]
            potentialIncomes.extend([x.income for x in household if x.status == 'retired'])
            potentialIncome = sum(potentialIncomes)
            
            house.residualIncomeForChildCare = potentialIncome
            house.initialResidualIncomeForChildCare = house.residualIncomeForChildCare
            
            # The household's tax brackets are the sum of the individual income brackets.
            taxBands = len(self.p['taxBrackets'])
            house.incomeByTaxBand = [0]*taxBands
            house.incomeByTaxBand[-1] = potentialIncome
            for i in range(taxBands-1):
                for income in potentialIncomes:
                    if income > self.p['taxBrackets'][i]:
                        bracket = income-self.p['taxBrackets'][i]
                        house.incomeByTaxBand[i] += bracket
                        house.incomeByTaxBand[-1] -= bracket
                        potentialIncomes[potentialIncomes.index(income)] -= bracket
                        

    def computeNetCareDemand(self):
        for house in self.map.occupiedHouses:
            house.suppliers = [house]
            house.suppliers.extend(list(house.careNetwork.neighbors(house)))
            house.receivers = list(house.demandNetwork.neighbors(house))
            totalDemand = house.totalUnmetSocialCareNeed + sum(house.childCareNeeds) # Total informal care need
            totalSupply = house.householdInformalSupplies[0]# Total internal care supply
            house.netCareDemand = totalDemand - totalSupply
            
                        
    def householdRelocation(self, policyFolder):
        shareRelocated = 0
        relocated = 0
        numHouseholds = len(self.map.occupiedHouses)
        for house in self.map.occupiedHouses:
            householdIncome = sum([x.income for x in house.occupants])
            house.incomePerCapita = householdIncome/float(len(house.occupants))
            house.careAttractionFactor = house.netCareDemand
        
        potentialMovers = [x for x in self.map.occupiedHouses if x.incomePerCapita > 0]
        for house in potentialMovers:
            house.townAttractiveness = []
            if house.newOccupancy == True:
                continue
            houseRank = max([x.classRank for x in house.occupants])
            perCapitaIncome = house.incomePerCapita
            house.sizeIndex = self.computeHouseholdDimension(house) - 1
            
            for town in self.map.towns:
                # First element: care-related factor (KNA)
                
                # If the household has positive net demand, the care attraction associated with a town depends on quantity of net supply within the 
                # household's care supply network.
                # Vice versa, if the household has positive net supply, the care attraction associated with a town depends on quantity of net supply within the
                # household's care demand network.
                
                if house.netCareDemand > 0:
                    neighbors = [x for x in house.suppliers if x != house and x.town == town and x.careAttractionFactor < 0]
                    distances = [house.careNetwork[house][x]['distance'] for x in neighbors]
                else:
                    neighbors = [x for x in house.receivers if x != house and x.town == town and x.careAttractionFactor > 0]
                    distances = [house.demandNetwork[house][x]['distance'] for x in neighbors]
                    
                cares = [x.careAttractionFactor for x in neighbors]
                # The total care demand/supply is weighted by the kinship distance
                totalCare = sum([abs(x)/math.pow(self.p['networkDistanceParam'], distances[cares.index(x)]) for x in cares])
#                for care in cares:
#                    distance = distances[cares.index(care)]
#                    totalCareWeight += abs(care)/math.pow(self.p['networkDistanceParam'], distance)
                
                # The total attraction depends on the product of the house care the network care: if the household has not net demand, ther is no use for
                # the network net supply, and vice versa,
                if house.netCareDemand > 0:
                    kinshipNetworkAttraction = (house.careAttractionFactor/math.exp(self.p['careAttractionExp']*perCapitaIncome))*totalCare
                else:
                    kinshipNetworkAttraction = abs(house.careAttractionFactor)*totalCare
                
                # First element of town attraction
                careFactor = math.exp(self.p['knaExp']*kinshipNetworkAttraction)
                
                
                # Second element: economic and social factors
                sameSES = len([x for x in town.houses if len(x.occupants) > 0 and max([y.classRank for y in x.occupants]) == houseRank])
                allHouses = len([x for x in town.houses if len(x.occupants) > 0])
                townSESShare = 0.0
                if allHouses > 0:
                    townSESShare = float(sameSES)/float(allHouses)
                sameSES = len([x for x in self.map.allHouses if len(x.occupants) > 0 and max([y.classRank for y in x.occupants]) == houseRank])
                allHouses = len([x for x in self.map.allHouses if len(x.occupants) > 0])
                mapSESShare = float(sameSES)/float(allHouses)
                deltaSES = townSESShare - mapSESShare
                sesFactor = math.exp(self.p['sesShareExp']*deltaSES)
                # relRent = house.town.LHA[house.sizeIndex]/(house.town.LHA[house.sizeIndex]+math.exp(self.p['relativeRentExp']*perCapitaIncome))
                relRent = house.town.LHA[house.sizeIndex]/perCapitaIncome
                rentFactor = 1/math.exp(self.p['rentExp']*relRent)
                
#                townHouses = len([x for x in town.houses if len(x.occupants) == 0])
#                mapHouses = len([x for x in self.map.allHouses if len(x.occupants) == 0])
#                shareAvailableHouses = float(townHouses)/float(mapHouses)
                sea = sesFactor*rentFactor
                
                # Town's total attractiveness (TTA)
                townTotalAttractiveness = careFactor*sea
                
                house.townAttractiveness.append(townTotalAttractiveness)
            
            # 1 - Sample a town other than the one where the family is living
            probTowns = []
            potentialTowns = []
            for town in self.map.towns:
                if town == house.town:
                    continue
                potentialTowns.append(town)
                indexTown = self.map.towns.index(town)
                townAttractiveness = house.townAttractiveness[indexTown]
                townHouses = len([x for x in town.houses if len(x.occupants) == 0])
                mapHouses = len([x for x in self.map.allHouses if len(x.occupants) == 0])
                shareAvailableHouses = float(townHouses)/float(mapHouses)
                probTowns.append(shareAvailableHouses*townAttractiveness)
            prob = [x/sum(probTowns) for x in probTowns]
            newTown = np.random.choice(potentialTowns, p = prob)
            
            # 2 - Decide whether to relocate to the new town or remain in the current one
            indexNewTown = self.map.towns.index(newTown)
            indexCurrentTown = self.map.towns.index(house.town)
            newTownAttractiveness = house.townAttractiveness[indexNewTown]
            currentTownAttractiveness = house.townAttractiveness[indexCurrentTown]
            attractivenessRatio = newTownAttractiveness/(newTownAttractiveness+currentTownAttractiveness)
            relocationCost = sum([math.pow(x.yearInTown, self.p['yearsInTownBeta']) for x in house.occupants])
            probRelocation = attractivenessRatio/math.exp(self.p['relocationCostBeta']*relocationCost)
            probRelocation *= self.p['scalingFactor']
            
            # print 'Prob Relocation: ' + str(probRelocation)
            
#                # Compute relocation probability
#                relocationCost = self.p['relocationCostParam']*sum([math.pow(x.yearInTown, self.p['yearsInTownBeta']) for x in household])
#                supportNetworkFactor = math.exp(self.p['supportNetworkBeta']*house.networkSupport)
#                relocationCostFactor = math.exp(self.p['relocationCostBeta']*relocationCost)
#                perCapitaIncome = self.computeHouseholdIncome(house)/float(len(household))
#                incomeFactor = math.exp(self.p['incomeRelocationBeta']*perCapitaIncome)
#                relativeRelocationFactor = (supportNetworkFactor*relocationCostFactor)/incomeFactor
#                probRelocation = self.p['baseRelocationRate']/relativeRelocationFactor
           
            if np.random.random() < probRelocation: #self.p['basicProbFamilyMove']*self.p['probFamilyMoveModifierByDecade'][int(ageClass)]:
                
                relocated += 1
                
                peopleToMove = [x for x in house.occupants]
                person = peopleToMove[0]
                
                if person.house == self.displayHouse:
                    messageString = str(self.year) + ": #" + str(person.id) + " and #" + str(person.partner.id) + " move house"
                    if len(peopleToMove) > 2:
                        messageString += " with kids"
                    messageString += "."

                self.findNewHouseInNewTown(peopleToMove, newTown)
        
        shareRelocated = float(relocated)/float(numHouseholds)
        print shareRelocated
        
    def computeHouseholdDimension(self, house):
        # Compute number of rooms for housing benefit purposes
        adults = [x for x in house.occupants if x.age > 15]
        houseBenefitRoom = 0
        visited = []
        for person in adults:
            if person in visited:
                continue
            visited.append(person)
            if person.partner != None:
                visited.append(person.partner)
            houseBenefitRoom +=1
        rooms = []
        children = [x for x in house.occupants if x.age < 16]
        children.sort(key=operator.attrgetter("age"), reverse = True)
        allocated = []
        for child in children:
            roomMates = []
            if child in allocated:
                continue
            allocated.append(child)
            roomMates.append(child)
            if child.age > 9:
                # Only another child of same sex can go with him
                sameSexChildren = [x for x in children if x.sex == child.sex and x.id != child.id]
                if len(sameSexChildren) > 0:
                    allocated.append(sameSexChildren[0])
                    roomMates.append(sameSexChildren[0])
            else:
                # Any child can share the room
                otherChildren = [x for x in children if x.id != child.id]
                if len(otherChildren) > 0:
                    allocated.append(otherChildren[0])
                    roomMates.append(otherChildren[0])
            rooms.append(roomMates)
        houseBenefitRoom += len(rooms)
        houseBenefitRoom = min(houseBenefitRoom, 4)
        houseBenefitRoom -= 1
        return houseBenefitRoom
        
    def doAgeTransitions(self):
        
        for person in self.pop.livingPeople:
            person.age += 1
            if person.age in self.p['ageBreaks']:
                person.ageClass += 1
            if person.age in self.p['interactionAgeBreaks']:
                person.interactionAgeClass += 1
            if person.age in self.p['mortalityAgeBreaks']:
                person.mortalityAgeClass += 1
            person.yearInTown += 1
            person.maternityStatus = False
        """Check whether people have moved on to a new status in life."""
        peopleNotYetRetired = [x for x in self.pop.livingPeople if x.status != 'retired']
        for person in peopleNotYetRetired:
            age = self.year - person.birthdate
            ## Do transitions to adulthood and retirement
            if person.age == self.p['ageTeenagers']:
                person.status == 'teenager'
            if person.age == self.p['ageOfAdulthood']:
                person.status = 'student'
                if np.random.random() < self.p['probOutOfTownStudent']:
                    person.outOfTownStudent = True
                person.classRank = 0 # max(person.father.classRank, person.mother.classRank)
                
            elif person.age == self.p['ageOfRetirement']:
                person.status = 'retired'

            ## If somebody is still at home but their parents have died, promote them to independent adult
            if person.mother != None:
                if person.status == 'worker' and person.mother not in person.house.occupants and person.father not in person.house.occupants:
                    person.independentStatus = True
            if person.status == 'student' and len([x for x in person.house.occupants if x.independentStatus == True]) == 0:
                if person.mother.dead:
                    if person.father.dead:
                        person.independentStatus = True
                        self.startWorking(person)
                    else:
                        self.movePeopleIntoChosenHouse(person.father.house, person.house,[person], 0)
                else:
                    self.movePeopleIntoChosenHouse(person.mother.house, person.house,[person], 0)
                    
            ## If somebody is a *child* at home and their parents have died, they need to be adopted
            if person.status == 'retired' and len([x for x in person.house.occupants if x.independentStatus == True]) == 0:
                person.independentStatus = True
            
            if person.status == 'child' and len([x for x in person.house.occupants if x.independentStatus == True]) == 0:
                if person.mother.dead:
                    if person.father.dead:
                
                        while True:
                            adoptiveMother = random.choice(self.pop.livingPeople)
                            if ( adoptiveMother.status != 'child'
                                 and adoptiveMother.sex == 'female'
                                 and adoptiveMother.partner != None ):
                                break
        
                        person.mother = adoptiveMother
                        adoptiveMother.children.append(person)
                        person.father = adoptiveMother.partner
                        adoptiveMother.partner.children.append(person)                

                        self.movePeopleIntoChosenHouse(adoptiveMother.house,person.house,[person], 0)   
                    else:
                        self.movePeopleIntoChosenHouse(person.father.house, person.house,[person], 0)
                else:
                    self.movePeopleIntoChosenHouse(person.mother.house, person.house,[person], 0)
                    
    def startWorking(self, person):
        
        person.status = 'worker'
        person.outOfTownStudent = False
        
        dKi = np.random.normal(0, self.p['wageVar'])
        person.initialIncome = self.p['incomeInitialLevels'][person.classRank]*math.exp(dKi)
        dKf = np.random.normal(dKi, self.p['wageVar'])
        person.finalIncome = self.p['incomeFinalLevels'][person.classRank]*math.exp(dKf)
        
        person.wage = person.initialIncome
        person.income = person.wage*self.p['dailyHours'][int(person.careNeedLevel)]
        person.potentialIncome = person.income
    
    def doSocialTransition(self):
        
        for person in self.pop.livingPeople:
            if person.age == self.p['workingAge'][person.classRank] and person.status == 'student':
            # With a certain probability p the person enters the workforce, 
            # with a probability 1-p goes to the next educational level
                if person.classRank == 4:
                    probStudy = 0.0
                else:
                    probStudy = self.transitionProb(person) # Probability of keeping studying
                
                if random.random() > probStudy:
                    self.startWorking(person)
                    
                else:
                    person.classRank += 1
                    
    def transitionProb (self, person):
        household = [x for x in person.house.occupants]
        if person.father.dead + person.mother.dead != 2:
            disposableIncome = sum([x.income for x in household])
            perCapitaDisposableIncome = disposableIncome/len(household)
            # print('Per Capita Disposable Income: ' + str(perCapitaDisposableIncome))
            
            if perCapitaDisposableIncome > 0.0:
                
                forgoneSalary = self.p['incomeInitialLevels'][person.classRank]*self.p['dailyHours'][person.careNeedLevel]
                educationCosts = self.p['educationCosts'][person.classRank]
                
                # relCost = (forgoneSalary+educationCosts)/perCapitaDisposableIncome
                
                relCost = forgoneSalary/perCapitaDisposableIncome
                
                # Check variable
#                if self.year == self.p['getCheckVariablesAtYear']:
#                    self.relativeEducationCost.append(relCost) # 0.2 - 5
                
                incomeEffect = (self.p['costantIncomeParam']+1)/(math.exp(self.p['eduWageSensitivity']*relCost) + self.p['costantIncomeParam']) # Min-Max: 0 - 10
                
                incomeFactor = math.exp(self.p['eduWageSensitivity']*perCapitaDisposableIncome)
                incomeEffect = incomeFactor/(incomeFactor+self.p['costantIncomeParam'])
                
                targetEL = max(person.father.classRank, person.mother.classRank)
                
                dE = float(targetEL - person.classRank)
                expEdu = math.exp(self.p['eduRankSensitivity']*dE)
                educationEffect = expEdu/(expEdu+self.p['costantEduParam'])
                
                careEffect = 1/math.exp(self.p['careEducationParam']*person.socialWork)
                
                
                ### Fixing probability to keep studying   ######################
                
                pStudy = incomeEffect*educationEffect*careEffect
                
#                shareAdjustmentFactor = self.socialClassShares[person.classRank] - self.p['shareClasses'][person.classRank]
#                
#                pStudy *= math.exp(self.p['classAdjustmentBeta']*shareAdjustmentFactor)
                
                if person.classRank == 0 and self.socialClassShares[0] > 0.2:
                    pStudy *= 1.0/0.9
                
                if person.classRank == 0 and self.socialClassShares[0] < 0.2:
                    pStudy *= 0.85
                
                if person.classRank == 1 and self.socialClassShares[1] > 0.35:
                    pStudy *= 1.0/0.8
                    
                if person.classRank == 2 and self.socialClassShares[2] > 0.25:
                    pStudy *= 1.0/0.85
                    
                
                #####################################################################
                
                # pStudy = math.pow(incomeEffect, self.p['incEduExp'])*math.pow(educationEffect, 1-self.p['incEduExp'])
                if pStudy < 0:
                    pStudy = 0
                # Check
#                if self.year == self.p['getCheckVariablesAtYear']:
#                    self.probKeepStudying.append(pStudy)
#                    self.person.classRankStudent.append(person.classRank)
                
            else:
                # print('perCapitaDisposableIncome: ' + str(perCapitaDisposableIncome))
                pStudy = 0
        else:
            pStudy = 0
        # pWork = math.exp(-1*self.p['eduEduSensitivity']*dE1)
        # return (pStudy/(pStudy+pWork))
        #pStudy = 0.8
        return pStudy
    
    
    def updateIncome(self):
        
        for person in self.pop.livingPeople:
            if person.status == 'worker' and person.careNeedLevel < 2:
                person.workExperience *= self.p['workDiscountingTime']
                if person.maternityStatus == False:
                    if self.p['dailyHours'][person.careNeedLevel] > 0:
                        person.workingPeriods += float(self.p['dailyHours'][person.careNeedLevel])/8.0
                        person.workExperience += person.residualWorkingHours/8.0
                    person.wage = self.computeWage(person)
                    person.income = person.wage*person.residualWorkingHours   # self.p['weeklyHours'][int(person.careNeedLevel)]
                    person.lastIncome = person.income
                elif person.maternityStatus == True:
                    person.wage = 0
                    person.income = 0
                   
            elif person.age == self.p['ageOfRetirement'] or person.careNeedLevel > 1:
                person.wage = 0
                shareWorkingTime = person.workingPeriods/float(self.p['minContributionYears'])
                dK = np.random.normal(0, self.p['wageVar'])
#                averageIncome = 0
#                if person.workingPeriods > 0:
#                    averageIncome = person.cumulativeIncome/person.workingPeriods
                person.income = person.lastIncome*shareWorkingTime*math.exp(dK) #self.p['pensionWage'][person.classRank]*self.p['weeklyHours'][0]
#                if person.income < self.p['statePension']:
#                    person.income = self.p['statePension']
            person.cumulativeIncome += (person.income - person.incomeExpenses)
            person.cumulativeIncome = max(person.cumulativeIncome, 0)
        
        self.grossDomesticProduct = sum([x.income for x in self.pop.livingPeople if x.wage > 0])
        
        for house in self.map.occupiedHouses:
            house.outOfWorkSocialCare = sum([x.outOfWorkSocialCare for x in house.occupants])
            house.householdIncome = sum([x.income for x in house.occupants])
            house.incomePerCapita = house.householdIncome/float(len(house.occupants))
           
        households = [x for x in self.map.occupiedHouses]
        # households.sort(key=operator.attrgetter("incomePerCapita"))
        households.sort(key=operator.attrgetter("householdIncome"))
        for i in range(5):
            number = int(round(len(households)/(5.0-float(i))))
            quintile = households[:number]
            for j in quintile:
                j.incomeQuintile = i
                for agent in j.occupants:
                    agent.incomeQuintile = i
            households = [x for x in households if x not in quintile]
 
        # Compute tax revenue and income after tax
        earningPeople = [x for x in self.pop.livingPeople if x.status == 'worker' and x.maternityStatus == False]
        self.totalTaxRevenue = 0
        self.totalPensionRevenue = 0
        for person in earningPeople:
            employeePensionContribution = 0
            # Pension Contributions
            if person.income > 162.0:
                if person.income < 893.0:
                    employeePensionContribution = (person.income - 162.0)*0.12
                else:
                    employeePensionContribution = (893.0 - 162.0)*0.12
                    employeePensionContribution += (person.income - 893.0)*0.02
            person.income -= employeePensionContribution
            self.totalPensionRevenue += employeePensionContribution
            
            # Tax Revenues
            tax = 0
            residualIncome = person.income
            for i in range(len(self.p['taxBrackets'])):
                if residualIncome > self.p['taxBrackets'][i]:
                    taxable = residualIncome - self.p['taxBrackets'][i]
                    tax += taxable*self.p['taxationRates'][i]
                    residualIncome -= taxable
            person.income -= tax
            self.totalTaxRevenue += tax
        self.statePensionRevenue.append(self.totalPensionRevenue)
        self.stateTaxRevenue.append(self.totalTaxRevenue)
        
        # Pensions Expenditure
        pensioners = [x for x in self.pop.livingPeople if x.status == 'retired']
        totalIncome = sum([x.income for x in earningPeople if x.status == 'worker'])
        self.pensionExpenditure = self.p['statePension']*len(pensioners) + totalIncome*self.p['statePensionContribution']
        self.statePensionExpenditure.append(self.pensionExpenditure)
        
        # Assign incomes according to empirical income distribution
        #####   Temporarily disactivating the top-down income attribution   ############################
        
#        earningPeople = [x for x in self.pop.livingPeople if x.income > 0] #x.status == 'worker']
#        earningPeople.sort(key=operator.attrgetter("income"))
#        
#        workersToAssign = list(earningPeople)
#        incomePercentiles = []
#        for i in range(99, 0, -1):
#            groupNum = int(float(len(workersToAssign))/float(i))
#            workersGroup = workersToAssign[0:groupNum]
#            incomePercentiles.append(workersGroup)
#            workersToAssign = workersToAssign[groupNum:]
#        
#        for i in range(99):
#            wage = float(self.incomesPercentiles[i])/(52*40)
#            for person in incomePercentiles[i]:
#                dK = np.random.normal(0, self.p['wageVar'])
#                person.wage = wage*math.exp(dK)
#                person.income = person.wage*self.p['weeklyHours'][int(person.careNeedLevel)]
        
    def updateWealth_Ind(self):
        # Only workers: retired are assigned a wealth at the end of their working life (which they consume thereafter)
        earningPop = [x for x in self.pop.livingPeople if x.cumulativeIncome > 0]
        
        earningPop.sort(key=operator.attrgetter("cumulativeIncome"))
        
        peopleToAssign = list(earningPop)
        wealthPercentiles = []
        for i in range(100, 0, -1):
            groupNum = int(float(len(peopleToAssign))/float(i))
            peopleGroup = peopleToAssign[0:groupNum]
            wealthPercentiles.append(peopleGroup)
            peopleToAssign = peopleToAssign[groupNum:]
            
        for i in range(100):
            wealth = float(self.wealthPercentiles[i])
            for person in wealthPercentiles[i]:
                dK = np.random.normal(0, self.p['wageVar'])
                person.wealth = wealth*math.exp(dK)
                
        for person in self.pop.livingPeople:
            # Update financial wealth
            if person.wage > 0:
                person.financialWealth = person.wealth*self.p['shareFinancialWealth']
            else:
                person.financialWealth -= person.wealthSpentOnCare
            person.financialWealth = max(person.financialWealth, 0)
        
        notEarningPop = [x for x in self.pop.livingPeople if x.cumulativeIncome > 0 and x.wage == 0]
        for person in notEarningPop:
            person.financialWealth *= (1.0 + self.p['pensionReturnRate'])
            
    def updateWealth(self):
        households = [x for x in self.map.occupiedHouses]
        for h in households:
            h.householdCumulativeIncome = sum([x.cumulativeIncome for x in h.occupants])
        households.sort(key=operator.attrgetter("householdCumulativeIncome"))
        
        householdsToAssign = list(households)
        wealthPercentiles = []
        for i in range(100, 0, -1):
            groupNum = int(float(len(householdsToAssign))/float(i))
            householdGroup = householdsToAssign[0:groupNum]
            wealthPercentiles.append(householdGroup)
            householdsToAssign = householdsToAssign[groupNum:]
            
        for i in range(100):
            wealth = float(self.wealthPercentiles[i])
            for household in wealthPercentiles[i]:
                dK = np.random.normal(0, self.p['wageVar'])
                household.wealth = wealth*math.exp(dK)
        
        # Assign household wealth to single members
        for h in households:
            if h.householdCumulativeIncome > 0:
                earningMembers = [x for x in h.occupants if x.cumulativeIncome > 0]
                for m in earningMembers:
                    m.wealth = (m.cumulativeIncome/h.householdCumulativeIncome)*h.wealth
            else:
                independentMembers = [x for x in h.occupants if x.independentStatus == True]
                if len(independentMembers) > 0:
                    for m in independentMembers:
                        m.wealth = h.wealth/float(len(independentMembers))
            
        
        
    def computeWage(self, person):
        
        # newK = self.p['incomeFinalLevels'][classRank]*math.exp(dK)    
        # c = np.math.log(self.p['incomeInitialLevels'][classRank]/newK)
        # wage = newK*np.math.exp(c*np.math.exp(-1*self.p['incomeGrowthRate'][classRank]*workExperience))
        c = np.math.log(person.initialIncome/person.finalIncome)
        wage = person.finalIncome*np.math.exp(c*np.math.exp(-1*self.p['incomeGrowthRate'][person.classRank]*person.workExperience))
        dK = np.random.normal(0, self.p['wageVar'])
        wage *= math.exp(dK)
        return wage
    
    def computeBirthProb(self, fertilityBias, rawRate, womanRank):
        womenOfReproductiveAge = [x for x in self.pop.livingPeople
                                  if x.sex == 'female' and x.age >= self.p['minPregnancyAge']]
        womanClassShares = []
        womanClassShares.append(len([x for x in womenOfReproductiveAge if x.classRank == 0])/float(len(womenOfReproductiveAge)))
        womanClassShares.append(len([x for x in womenOfReproductiveAge if x.classRank == 1])/float(len(womenOfReproductiveAge)))
        womanClassShares.append(len([x for x in womenOfReproductiveAge if x.classRank == 2])/float(len(womenOfReproductiveAge)))
        womanClassShares.append(len([x for x in womenOfReproductiveAge if x.classRank == 3])/float(len(womenOfReproductiveAge)))
        womanClassShares.append(len([x for x in womenOfReproductiveAge if x.classRank == 4])/float(len(womenOfReproductiveAge)))
        a = 0
        for i in range(int(self.p['numberClasses'])):
            a += womanClassShares[i]*math.pow(self.p['fertilityBias'], i)
        baseRate = rawRate/a
        birthProb = baseRate*math.pow(self.p['fertilityBias'], womanRank)
        return birthProb
    
    def doBirths(self):
        
        preBirth = len(self.pop.livingPeople)
        
        marriedLadies = 0
        adultLadies = 0
        births = [0, 0, 0, 0, 0]
        marriedPercentage = []
        
        adultWomen = [x for x in self.pop.livingPeople
                                       if x.sex == 'female' and x.age >= self.p['minPregnancyAge']]
        
        womenOfReproductiveAge = [x for x in self.pop.livingPeople
                                  if x.sex == 'female'
                                  and x.age >= self.p['minPregnancyAge']
                                  and x.age <= self.p['maxPregnancyAge']
                                  and x.partner != None]
        
        adultLadies_1 = [x for x in adultWomen if x.classRank == 0]   
        marriedLadies_1 = len([x for x in adultLadies_1 if x.partner != None])
        if len(adultLadies_1) > 0:
            marriedPercentage.append(marriedLadies_1/float(len(adultLadies_1)))
        else:
            marriedPercentage.append(0)
        adultLadies_2 = [x for x in adultWomen if x.classRank == 1]    
        marriedLadies_2 = len([x for x in adultLadies_2 if x.partner != None])
        if len(adultLadies_2) > 0:
            marriedPercentage.append(marriedLadies_2/float(len(adultLadies_2)))
        else:
            marriedPercentage.append(0)
        adultLadies_3 = [x for x in adultWomen if x.classRank == 2]   
        marriedLadies_3 = len([x for x in adultLadies_3 if x.partner != None]) 
        if len(adultLadies_3) > 0:
            marriedPercentage.append(marriedLadies_3/float(len(adultLadies_3)))
        else:
            marriedPercentage.append(0)
        adultLadies_4 = [x for x in adultWomen if x.classRank == 3]  
        marriedLadies_4 = len([x for x in adultLadies_4 if x.partner != None])   
        if len(adultLadies_4) > 0:
            marriedPercentage.append(marriedLadies_4/float(len(adultLadies_4)))
        else:
            marriedPercentage.append(0)
        adultLadies_5 = [x for x in adultWomen if x.classRank == 4]   
        marriedLadies_5 = len([x for x in adultLadies_5 if x.partner != None]) 
        if len(adultLadies_5) > 0:
            marriedPercentage.append(marriedLadies_5/float(len(adultLadies_5)))
        else:
            marriedPercentage.append(0)
        
        # print(marriedPercentage)
        
#        for person in self.pop.livingPeople:
#           
#            if person.sex == 'female' and person.age >= self.p['minPregnancyAge']:
#                adultLadies += 1
#                if person.partner != None:
#                    marriedLadies += 1
#        marriedPercentage = float(marriedLadies)/float(adultLadies)
        
        for woman in womenOfReproductiveAge:
            
            womanClassRank = woman.classRank
            if woman.status == 'student':
                womanClassRank = woman.parentsClassRank

            if self.year < 1951:
                rawRate = self.p['growingPopBirthProb']
                birthProb = self.computeBirthProb(self.p['fertilityBias'], rawRate, womanClassRank)
            else:
                rawRate = self.fert_data[(self.year - woman.birthdate)-16, self.year-1950]
                birthProb = self.computeBirthProb(self.p['fertilityBias'], rawRate, womanClassRank)/marriedPercentage[womanClassRank]
                
            # birthProb = self.computeBirthProb(self.p['fertilityBias'], rawRate, woman.classRank)
            
            #baseRate = self.baseRate(self.socialClassShares, self.p['fertilityBias'], rawRate)
            #fertilityCorrector = (self.socialClassShares[woman.classRank] - self.p['initialClassShares'][woman.classRank])/self.p['initialClassShares'][woman.classRank]
            #baseRate *= 1/math.exp(self.p['fertilityCorrector']*fertilityCorrector)
            #birthProb = baseRate*math.pow(self.p['fertilityBias'], woman.classRank)
            
            if random.random() < birthProb:
                # (self, mother, father, age, birthYear, sex, status, house,
                # classRank, sec, edu, wage, income, finalIncome):
                parentsClassRank = max([woman.classRank, woman.partner.classRank])
                baby = Person(woman, woman.partner, self.year, 0, 'random', woman.house, woman.sec, -1, parentsClassRank, 
                              0, 0, 0, 0, 0, 0, 'child', False, 'susceptible')
                baby.lifeExpectancy = max(90-baby.age, 5.0)
                self.pop.allPeople.append(baby)
                self.pop.livingPeople.append(baby)
                woman.house.occupants.append(baby)
                woman.children.append(baby)
                woman.partner.children.append(baby)
                woman.maternityStatus = True
                woman.residualWorkingHours = 0
                woman.availableWorkingHours = 0
                woman.potentialIncome = 0
                woman.income = 0
                    
        postBirth = len(self.pop.livingPeople)
        
        print('the number of births is: ' + str(postBirth - preBirth))
    
    def computeSplitProb(self, rawRate, classRank):
        a = 0
        for i in range(int(self.p['numberClasses'])):
            a += self.socialClassShares[i]*math.pow(self.p['divorceBias'], i)
        baseRate = rawRate/a
        splitProb = baseRate*math.pow(self.p['divorceBias'], classRank)
        return splitProb
            
    def doDivorces(self):
        menInRelationships = [x for x in self.pop.livingPeople if x.sex == 'male' and x.partner != None ]
        for man in menInRelationships:
            
            age = self.year - man.birthdate 

            if self.year < self.p['thePresent']:
                rawRate = self.p['basicDivorceRate'] * self.p['divorceModifierByDecade'][int(age)/10]
            else:
                rawRate = self.p['variableDivorce'] * self.p['divorceModifierByDecade'][int(age)/10]
                
            splitProb = self.computeSplitProb(rawRate, man.classRank)
                
            if random.random() < splitProb:
                wife = man.partner
                man.partner = None
                wife.partner = None
                man.yearDivorced.append(self.year)
                wife.yearDivorced.append(self.year)
                if wife.status == 'student':
                    wife.independentStatus = True
                    self.startWorking(wife)
                self.divorceTally += 1
                towns = [man.house.town]
                # manChildren = [x for x in man.children if x.dead == False and x.house == man.house and x.father == man and x.mother != wife]
                
                manChildren = []
                children = [x for x in man.children if x.dead == False and x.house == man.house]
                for child in children:
                    if child.father == man and child.mother != wife:
                        manChildren.append(child)
                    else:
                        if np.random.random() < self.p['probChildrenWithFather']:
                            manChildren.append(child)
                        
                peopleToMove = [man]
                peopleToMove += manChildren
                self.findNewHouse(peopleToMove, towns)
                
    def doMarriages(self):
        
        eligibleMen = [x for x in self.pop.livingPeople if x.sex == 'male' and x.partner == None and x.status != 'child' and x.status != 'student']
        eligibleWomen = [x for x in self.pop.livingPeople if x.sex == 'female' and x.partner == None and x.status != 'child']
        
        
        if len(eligibleMen) > 0 and len (eligibleWomen) > 0:
            random.shuffle(eligibleMen)
            random.shuffle(eligibleWomen)
            
            interestedWomen = []
            for w in eligibleWomen:
                womanMarriageProb = self.p['basicFemaleMarriageProb']*self.p['femaleMarriageModifierByDecade'][w.age/10]
                if random.random() < womanMarriageProb:
                    interestedWomen.append(w)
        
            for man in eligibleMen:
                
                ageClass = int(man.age/10)
                manMarriageProb = self.p['basicMaleMarriageProb']*self.p['maleMarriageModifierByDecade'][ageClass]
                
                ageClassPop = [x for x in eligibleMen if int(x.age/10) == ageClass]
                noChildrenMen = [x for x in ageClassPop if len([y for y in x.children if y.dead == False and y.house == x.house]) == 0]
                shareNoChildren = float(len(noChildrenMen))/float(len(ageClassPop))
                den = shareNoChildren + (1-shareNoChildren)*self.p['manWithChildrenBias']
                baseRate = manMarriageProb/den
                if man in noChildrenMen:
                    manMarriageProb = baseRate
                else:
                    manMarriageProb = baseRate*self.p['manWithChildrenBias']

                manMarriageProb *= self.p['maleMarriageMultiplier']
                
                
                if random.random() < manMarriageProb:
                    potentialBrides = []
                    for woman in interestedWomen:
                        deltaAge = man.age - woman.age
                        if deltaAge < 20 and deltaAge > -10:
                            if woman.house != man.house:
                                if man.mother != None and woman.mother != None:
                                    if man.mother != woman.mother and man not in woman.children and woman not in man.children:
                                        potentialBrides.append(woman)
                                else:
                                    if man not in woman.children and woman not in man.children:
                                        potentialBrides.append(woman)
                    
                    if len(potentialBrides) > 0:
                        manTown = man.house.town
                        bridesWeights = []
                        for woman in potentialBrides:
                            
                            womanTown = woman.house.town
                            geoDistance = self.manhattanDistance(manTown, womanTown)/float(self.p['mapGridXDimension'] + self.p['mapGridYDimension'])
                            geoFactor = 1/math.exp(self.p['betaGeoExp']*geoDistance)
                            
                            womanRank = woman.classRank
                            studentFactor = 1.0
                            if woman.status == 'student':
                                studentFactor = self.p['studentFactorParam']
                                womanRank = max(woman.father.classRank, woman.mother.classRank)
                            statusDistance = float(abs(man.classRank-womanRank))/float((self.p['numberClasses']-1))
                            if man.classRank < womanRank:
                                betaExponent = self.p['betaSocExp']
                            else:
                                betaExponent = self.p['betaSocExp']*self.p['rankGenderBias']
                            socFactor = 1/math.exp(betaExponent*statusDistance)
                            
                            ageFactor = self.p['deltageProb'][self.deltaAge(man.age-woman.age)]
                            
                            numChildrenWithWoman = len([x for x in woman.children if x.house == woman.house])
                            
                            childrenFactor = 1/math.exp(self.p['bridesChildrenExp']*numChildrenWithWoman)
                            
                            marriageProb = geoFactor*socFactor*ageFactor*childrenFactor*studentFactor
                            
                            bridesWeights.append(marriageProb)
                            
                        if sum(bridesWeights) > 0:
                            bridesProb = [i/sum(bridesWeights) for i in bridesWeights]
                            woman = np.random.choice(potentialBrides, p = bridesProb)
                        else:
                            woman = np.random.choice(potentialBrides)
                        man.partner = woman
                        woman.partner = man
                        man.yearMarried.append(self.year)
                        woman.yearMarried.append(self.year)
                        interestedWomen.remove(woman)
                        
                        self.marriageTally += 1
 
    def deltaAge(self, dA):
        if dA <= -10 :
            cat = 0
        elif dA >= -9 and dA <= -3:
            cat = 1
        elif dA >= -2 and dA <= 0:
            cat = 2
        elif dA >= 1 and dA <= 4:
            cat = 3
        elif dA >= 5 and dA <= 9:
            cat = 4
        else:
            cat = 5
        return cat

    
    def doMovingAround(self):
        """
        Various reasons why a person or family group might want to
        move around. People who are in partnerships but not living
        together are highly likely to find a place together. Adults
        still living at home might be ready to move out this year.
        Single people might want to move in order to find partners. A
        family might move at random for work reasons. Older people
        might move back in with their kids.
        """
        self.relocationCostFactors[:] = []
        self.incomeFactors[:] = []
        self.probsRelocation[:] = []
        
        for i in self.pop.livingPeople:
            i.movedThisYear = False
            
        ## Need to keep track of this to avoid multiple moves
        separetedSpouses = [x for x in self.pop.livingPeople if x.partner != None and x.house != x.partner.house]
        couples = []
        for i in separetedSpouses:
            couples.append([i, i.partner])
       
        for person in separetedSpouses:
            
            if person.house != person.partner.house:
                if random.random() < self.p['probApartWillMoveTogether']:
                    peopleToMove = [person, person.partner]
                    personChildren = self.bringTheKids(person)
                    personChildrenToMove = [x for x in personChildren if x not in separetedSpouses]
                    peopleToMove += personChildrenToMove
                    partnerChildren = self.bringTheKids(person.partner)
                    partnerChildrenToMove = [x for x in partnerChildren if x not in separetedSpouses]
                    peopleToMove += [x for x in partnerChildrenToMove if x not in personChildrenToMove]
                    
                    if random.random() < self.p['coupleMovesToExistingHousehold']:
                        myHousePop = len(person.house.occupants)
                        yourHousePop = len(person.partner.house.occupants)
                        if yourHousePop < myHousePop:
                            targetHouse = person.partner.house
                        else:
                            targetHouse = person.house
                                
                        self.movePeopleIntoChosenHouse(targetHouse,person.house,peopleToMove, 0)                        
                    else:
                        
                        towns = [person.house.town, person.partner.house.town]
                                
                        self.findNewHouse(peopleToMove, towns)                        
    
                    if person.independentStatus == False:
                        person.independentStatus = True
                    if person.partner.independentStatus == False:
                        person.partner.independentStatus = True

        for person in self.pop.livingPeople:
            age = self.year - person.birthdate
            ageClass = age / 10
            
            if person.movedThisYear:
                continue
            
            elif person.status == 'worker' and person.independentStatus == False and person.partner == None:
                ## a single person who hasn't left home yet
                if random.random() < ( self.p['basicProbAdultMoveOut']
                                       * self.p['probAdultMoveOutModifierByDecade'][ageClass] ):
                    peopleToMove = [person]
                    peopleToMove += self.bringTheKids(person)
                    distance = random.choice(['here','near'])
                    
                    # Select a town based on:
                    # 1 - Distance form current town
                    # 2 - Size of town
                    # Gravitational model: town attraction = Size/pow(distance+1, beta)
                    towns = self.selectTown(person.house.town)    #[person.house.town]
                    
#                    if distance == 'near':
#                        nearbyTowns = [ k for k in self.map.towns if abs(k.x - town.x) <= 1 and abs(k.y - town.y) <= 1 ]
#                        weights = [float(len(x.houses)) for x in nearbyTowns]
#                        probs = [x/sum(weights) for x in weights]
#                        town = np.random.choice(nearbyTowns, p = probs)
                            
                    self.findNewHouse(peopleToMove, towns)
                    person.independentStatus = True
                    

#            elif person.independentStatus == True and person.partner == None:
#                ## a young-ish person who has left home but is still (or newly) single
#                if random.random() < ( self.p['basicProbSingleMove']
#                                       * self.p['probSingleMoveModifierByDecade'][int(ageClass)] ):
#                    peopleToMove = [person]
#                    peopleToMove += self.bringTheKids(person)
#                    distance = random.choice(['here','near'])
#                    if person.house == self.displayHouse:
#                        messageString = str(self.year) + ": #" + str(person.id) + " moves to meet new people."
#                        self.textUpdateList.append(messageString)
#                        with open(os.path.join(policyFolder, "Log.csv"), "a") as file:
#                            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
#                            writer.writerow([self.year, messageString])
#                            
#                    self.findNewHouse(peopleToMove,distance, policyFolder)

            elif person.status == 'retired' and len(person.house.occupants) == 1:
                ## a retired person who lives alone
                for c in person.children:
                    if ( c.dead == False ):
                        distance = self.manhattanDistance(person.house.town,c.house.town)
                        distance += 1.0
                        if self.year < self.p['thePresent']:
                            mbRate = self.p['agingParentsMoveInWithKids']/distance
                        else:
                            mbRate = self.p['variableMoveBack']/distance
                        if random.random() < mbRate:
                            peopleToMove = [person]
                                
                            self.movePeopleIntoChosenHouse(c.house,person.house,peopleToMove, 0)
                            break
                        
        
            
            elif person.partner != None and person.yearMarried[-1] != self.year and person.house.vacated == False:
                ## any other kind of married person, e.g., a family with kids
                house = person.house
                household = [x for x in house.occupants]
                
                # Compute relocation probability
                # Relocation cost: the higher the lower the prob to relocate
                relocationCost = self.p['relocationCostParam']*sum([math.pow(x.yearInTown, self.p['yearsInTownBeta']) for x in household])
                relocationCostFactor = 1.0/math.exp(self.p['relocationCostBeta']*relocationCost)
                # Support the household is getting in the current town: the hoigher the lower the probability to relocate
                supportNetworkFactor = 1.0/math.exp(self.p['supportNetworkBeta']*house.networkSupport)
                # Income factor: the higher the higher the probability to relocate
                perCapitaIncome = self.computeHouseholdIncome(house)/float(len(household))
                incomeFactorElement = math.exp(self.p['incomeRelocationBeta']*perCapitaIncome)
                incomeFactor = (incomeFactorElement-1.0)/incomeFactorElement
                relativeRelocationFactor = supportNetworkFactor*relocationCostFactor*incomeFactor
                probRelocation = self.p['baseRelocationRate']*relativeRelocationFactor
                
                self.relocationCostFactors.append(relocationCostFactor)
                self.incomeFactors.append(incomeFactor)
                self.probsRelocation.append(probRelocation)
                if np.random.random() < probRelocation: #self.p['basicProbFamilyMove']*self.p['probFamilyMoveModifierByDecade'][int(ageClass)]:
                    person.house.vacated = True
                    peopleToMove = [x for x in person.house.occupants]
#                    personChildren = self.bringTheKids(person)
#                    peopleToMove += personChildren
#                    partnerChildren = self.bringTheKids(person.partner)
#                    peopleToMove += [x for x in partnerChildren if x not in personChildren]
#                    stepChildrenPartner = [x for x in personChildren if x not in partnerChildren]
#                    stepChildrenPerson = [x for x in partnerChildren if x not in personChildren]
#                    person.children.extend(stepChildrenPerson)
#                    person.partner.children.extend(stepChildrenPartner)
                    
                    # Add a choice of town which depends on kinship network and available houses.
                    
                    # distance = random.choice(['here,''near','far'])
                    towns = self.selectTown(person.house.town)
                    if person.house == self.displayHouse:
                        messageString = str(self.year) + ": #" + str(person.id) + " and #" + str(person.partner.id) + " move house"
                        if len(peopleToMove) > 2:
                            messageString += " with kids"
                        messageString += "."
                        self.textUpdateList.append(messageString)
                        with open(os.path.join(policyFolder, "Log.csv"), "a") as file:
                            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
                            writer.writerow([self.year, messageString])
                        
                    self.findNewHouse(peopleToMove,towns)
                    
        
        # Update display house
        if len(self.displayHouse.occupants) < 1:
            self.displayHouse.display = False
            ## Nobody lives in the display house any more, choose another
            if self.nextDisplayHouse != None:
                self.displayHouse = self.nextDisplayHouse
                self.displayHouse.display = True
                self.nextDisplayHouse = None
            else:
                self.displayHouse = random.choice(self.pop.livingPeople).house
                self.displayHouse.display = True
                self.textUpdateList.append(str(self.year) + ": Display house empty, going to " + self.displayHouse.name + ".")
                messageString = "Residents: "
                for k in self.displayHouse.occupants:
                    messageString += "#" + str(k.id) + " "
                self.textUpdateList.append(messageString)
    
    def selectTown(self, personTown):
        sizes = [len(x.houses) for x in self.map.towns]
        distances = [self.manhattanDistance(personTown, x) for x in self.map.towns]
        weights = [float(s)/float(np.power(d+1, self.p['townSelectionExp'])) for s, d, in zip(sizes, distances)]
        probs = [x/sum(weights) for x in weights]
        newTown = np.random.choice(self.map.towns, p = probs)
        return [newTown]
    
    def checkHouseholdsRelocations(self):

        newResidences = len([x for x in self.map.occupiedHouses if x.vacated == True])
        shareRelocations = float(newResidences)/float(len(self.map.occupiedHouses))
        
        print 'Mean relocation cost factors: ' + str(np.mean(self.relocationCostFactors))
        print 'Mean income factors: ' + str(np.mean(self.incomeFactors))
        print 'Mean prob relocation: ' + str(np.mean(self.probsRelocation))
        print 'Share of households relocated: ' + str(shareRelocations)
        
        for house in self.map.allHouses:
            house.vacated = False      
    
    def selectSpousesTown(self, town1, town2):
        prob1 = float(len(town1.houses))/(float(len(town1.houses))+float(len(town2.houses)))
        if np.random.random() < prob1:
            return town1
        else:
            return town2
    
    def manhattanDistance(self,t1,t2):
        """Calculates the distance between two towns"""
        xDist = abs(t1.x - t2.x)
        yDist = abs(t1.y - t2.y)
        return xDist + yDist


    def bringTheKids(self,person):
        """Given a person, return a list of their dependent kids who live in the same house as them."""
        returnList = []
        for i in person.children:
            if ( i.house == person.house
                 and i.independentStatus == False
                 and i.dead == False ):
                returnList.append(i)
        return returnList

    def findNewHouseInNewTown(self, personList, newTown):
        """Find an appropriate empty house for the named person and put them in it."""

        person = personList[0]
        departureHouse = person.house
        maxQuintile = max([x.incomeQuintile for x in person.house.occupants])
        
        emptyHousesInTown = []
        houseDesirabilityIndex = []
        rearrangedAvailableHouses = []
        availableHouses = [x for x in self.map.allHouses if x.town == newTown and len(x.occupants) < 1]
        emptyHousesInTown.extend(availableHouses)
        occupiedHouses = [x for x in self.map.allHouses if x.town == newTown and len(x.occupants) > 0]
        if len(occupiedHouses) > 0:
            for house in availableHouses:
                rearrangedAvailableHouses.append(house)
                affinityIndexes = []
                for unit in occupiedHouses:
                    distance = self.geoDistance(house, unit)
                    deltaClass = float(abs(unit.occupants[0].incomeQuintile-maxQuintile))
                    affinity = 1.0/math.exp(self.p['classAffinityExp']*deltaClass)
                    affinityIndexes.append(affinity/math.pow(distance, self.p['distanceAffinityExp']))
                houseDesirabilityIndex.append(np.mean(affinityIndexes))
        if sum(houseDesirabilityIndex) > 0:
            probs = [x/sum(houseDesirabilityIndex) for x in houseDesirabilityIndex]
            newHouse = np.random.choice(rearrangedAvailableHouses, p = probs)
        else:
            if len(emptyHousesInTown) > 0:
                newHouse = np.random.choice(emptyHousesInTown)
            else:
                newHouse = np.random.choice([x for x in self.map.allHouses if len(x.occupants) < 1])

        ## Quit with an error message if we've run out of houses
        if newHouse in self.map.occupiedHouses:
            print 'Error in house selection: already occupied!'
            print newHouse.id
            
#        if newHouse == None:
#            print "No houses left for person of SEC " + str(person.sec)
#            sys.exit()

        ## Actually make the chosen move
        self.movePeopleIntoChosenHouse(newHouse, departureHouse, personList, 1)

    def findNewHouse(self, personList, towns):
        ###  Called from:
        ###  - divorce function: husband must find a new house
        ###  - doMovingAround function
        
        """Find an appropriate empty house for the named person and put them in it."""
        ###  Change this function: find a house with probabilities which depends on the house attractivity
        ### House attractivities depend on SES
        
        person = personList[0]
        departureHouse = person.house
        
        emptyHousesInTown = []
        houseDesirabilityIndex = []
        rearrangedAvailableHouses = []
        for town in towns:
            availableHouses = [x for x in self.map.allHouses if x.town == town and len(x.occupants) < 1]
            emptyHousesInTown.extend(availableHouses)
            occupiedHouses = [x for x in self.map.allHouses if x.town == town and len(x.occupants) > 0]
            if len(occupiedHouses) > 0:
                for house in availableHouses:
                    rearrangedAvailableHouses.append(house)
                    affinityIndexes = []
                    for unit in occupiedHouses:
                        distance = self.geoDistance(house, unit)
                        deltaClass = float(abs(unit.occupants[0].incomeQuintile-personList[0].incomeQuintile))
                        affinity = 1.0/math.exp(self.p['classAffinityExp']*deltaClass)
                        affinityIndexes.append(affinity/math.pow(distance, self.p['distanceAffinityExp']))
                    houseDesirabilityIndex.append(np.mean(affinityIndexes))
        if sum(houseDesirabilityIndex) > 0:
            probs = [x/sum(houseDesirabilityIndex) for x in houseDesirabilityIndex]
            newHouse = np.random.choice(rearrangedAvailableHouses, p = probs)
        else:
            if len(emptyHousesInTown) > 0:
                newHouse = np.random.choice(emptyHousesInTown)
            else:
                newHouse = np.random.choice([x for x in self.map.allHouses if len(x.occupants) < 1])
        
        # chose a new house 
#        newHouse = None

#        t = person.house.town
#        if preference == 'here':
#            ## Anything empty in this town of the right size?
#            localPossibilities = [x for x in t.houses
#                                  if len(x.occupants) < 1
#                                  and person.sec == x.size ]
#            if localPossibilities:
#                newHouse = random.choice(localPossibilities)
#        if preference == 'near' or newHouse == None:
#            ## Neighbouring towns?
#            if newHouse == None:
#                nearbyTowns = [ k for k in self.map.towns
#                                if abs(k.x - t.x) <= 1
#                                and abs(k.y - t.y) <= 1 ]
#                nearbyPossibilities = []
#                for z in nearbyTowns:
#                    for w in z.houses:
#                        if len(w.occupants) < 1 and person.sec == w.size:
#                            nearbyPossibilities.append(w)
#                if nearbyPossibilities:
#                    newHouse = random.choice(nearbyPossibilities)
#        if ( preference == 'far' or newHouse == None ):
#            ## Anywhere at all?
#            if newHouse == None:
#                allPossibilities = []
#                for z in self.map.allHouses:
#                    if len(z.occupants) < 1 and person.sec == z.size:
#                        allPossibilities.append(z)
#                if allPossibilities:
#                    newHouse = random.choice(allPossibilities)

        ## Quit with an error message if we've run out of houses
        if newHouse in self.map.occupiedHouses:
            print 'Error in house selection: already occupied!'
            print newHouse.id
            
#        if newHouse == None:
#            print "No houses left for person of SEC " + str(person.sec)
#            sys.exit()

        ## Actually make the chosen move
        self.movePeopleIntoChosenHouse(newHouse,departureHouse,personList, 1)


    def movePeopleIntoChosenHouse(self,newHouse,departureHouse,personList, case):

        ## Put the new house onto the list of occupied houses if it was empty
        household = list(personList)
        
        newHouse.newOccupancy = True
        
        if len(household) != len(set(household)):
            print 'Error in movePeopleIntoChosenHouse: double counting people'
            for member in household:
                print member.id
            sys.exit()
        
        
        ## Move everyone on the list over from their former house to the new one
        for i in household:
            if newHouse.town != departureHouse.town:
                i.yearInTown = 0
                
            oldHouse = i.house
                
            oldHouse.occupants.remove(i)
            
            if len(oldHouse.occupants) == 0:
                self.map.occupiedHouses.remove(oldHouse)
                ##print "This house is now empty: ", oldHouse
                if (self.p['interactiveGraphics']):
                    self.canvas.itemconfig(oldHouse.icon, state='hidden')
            
            newHouse.occupants.append(i)
            
            i.house = newHouse
            i.movedThisYear = True

        if case == 1:
            self.map.occupiedHouses.append(newHouse)

            
    
                

    def doStats(self, day, policyFolder, dataMapFolder, dataHouseholdFolder):
        """Calculate annual stats and store them appropriately."""
        
        self.times.append(day)

        currentPop = len(self.pop.livingPeople)
        everLivedPop = len(self.pop.allPeople)
        self.pops.append(currentPop)
         
        
        potentialWorkers = [x for x in self.pop.livingPeople if x.age >= self.p['ageOfAdulthood'] and x.age < self.p['ageOfRetirement']]
        employed = [x for x in potentialWorkers if x.status == 'worker' and x.residualWorkingHours > 0]
        potentialHours = sum([self.p['dailyHours'][x.careNeedLevel] for x in employed])
        actualHours = sum([x.residualWorkingHours for x in employed])
        
        shareEmployed = 0
        if len(potentialWorkers) > 0:
            shareEmployed = float(len(employed))/float(len(potentialWorkers))
        shareWorkHours = 0
        if potentialHours > 0:
            shareWorkHours = float(actualHours)/float(potentialHours)
        self.employmentRate.append(shareEmployed)
        self.shareWorkingHours.append(shareWorkHours)

        ## Check for double-included houses by converting to a set and back again
        self.map.occupiedHouses = list(set(self.map.occupiedHouses))

        parents = []
        for person in self.pop.livingPeople:
            for child in person.children:
                if person.house == child.house and person.partner != None and (child.age <= 15 or (child.age <= 18 and child.status == 'student')): # self.p['maxWtWChildAge']:
                    parents.append(person)
                    break
        numberCouples = float(len(parents))/2
        loneParents = []
        loneFemaleParents = []
        for person in self.pop.livingPeople:
            for child in person.children:
                if person.house == child.house and person.partner == None and (child.age <= 15 or (child.age <= 18 and child.status == 'student')): # self.p['maxWtWChildAge']:
                    loneParents.append(person)
                    if person.sex == 'female':
                        loneFemaleParents.append(person)
                    break
        numberLoneParents = float(len(loneParents))
        if (numberCouples+numberLoneParents) > 0:
            shareSingleParents = numberLoneParents/(numberCouples+numberLoneParents)
        shareFemaleSingleParent = 0
        if numberLoneParents > 0:
            shareFemaleSingleParent = float(len(loneFemaleParents))/numberLoneParents
        self.shareLoneParents.append(shareSingleParents)
        self.shareFemaleLoneParents.append(shareFemaleSingleParent)
        
        over64 = [x for x in self.pop.livingPeople if x.age >= 65]
        indipendentOver65 = [x for x in over64 if x.careNeedLevel == 0]
        lowDependencyOver65 = [x for x in over64 if x.careNeedLevel == 1]
        mediumDependencyOver65 = [x for x in over64 if x.careNeedLevel == 2 or x.careNeedLevel == 3]
        highDependencyOver65 = [x for x in over64 if x.careNeedLevel == 4]
        
#        if len(over64) > 0:
#            print 'independent Over 64: ' + str(float(len(indipendentOver65))/float(len(over64)))
#            print 'low dependency Over 64: ' + str(float(len(lowDependencyOver65))/float(len(over64)))
#            print 'medium dependency Over 64: ' + str(float(len(mediumDependencyOver65))/float(len(over64)))
#            print 'high dependency Over 64: ' + str(float(len(highDependencyOver65))/float(len(over64)))
#        
        if self.year == 2017:
            self.sesPops = []
            for i in range(int(self.p['numberClasses'])):
                self.sesPops.append(len([x for x in self.pop.livingPeople if x.age > 23 and x.classRank == i]))
            self.sesPopsShares = [float(x)/float(sum(self.sesPops)) for x in self.sesPops]
            lenFrequency = len(self.incomeDistribution)
            self.incomeFrequencies = [0]*lenFrequency
            households = [y.occupants for y in self.map.occupiedHouses]
            
            self.individualIncomes = [x.income*52 for x in self.pop.livingPeople if x.income > 0]
            
        
            self.householdIncomes = [sum([x.income*52 for x in y]) for y in households]
            
            for i in self.householdIncomes:
                ind = int(i/1000)
                if ind > -1 and ind < lenFrequency:
                    self.incomeFrequencies[ind] += 1

        ## Check for overlooked empty houses
        emptyHouses = [x for x in self.map.occupiedHouses if len(x.occupants) == 0]
        for h in emptyHouses:
            self.map.occupiedHouses.remove(h)
            if (self.p['interactiveGraphics']):
                self.canvas.itemconfig(h.icon, state='hidden')

        ## Avg household size (easily calculated by pop / occupied houses)
        numHouseholds = len(self.map.occupiedHouses)
        averageHouseholdSize = 0
        if float(numHouseholds) > 0:
            averageHouseholdSize = float(currentPop)/float(numHouseholds)
        self.avgHouseholdSize.append(averageHouseholdSize)

        self.numMarriages.append(self.marriageTally)
        self.numDivorces.append(self.divorceTally)            
        
        
        totalSocialCareNeed = sum([x.hoursSocialCareDemand for x in self.pop.livingPeople])
        totalInformalSocialCare = sum([x.informalSocialCareReceived for x in self.pop.livingPeople])
        totalFormalSocialCare = sum([x.formalSocialCareReceived for x in self.pop.livingPeople])
        totalSocialCare = totalInformalSocialCare + totalFormalSocialCare
        totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in self.pop.livingPeople])
        shareInHouseSocialCare = 0
        if totalInformalSocialCare > 0:
            shareInHouseSocialCare = self.inHouseInformalCare/totalInformalSocialCare
        
        print 'Share of in-house informal care: ' + str(shareInHouseSocialCare)
        
        share_InformalSocialCare = 0
        if totalSocialCare > 0:
            share_InformalSocialCare = totalInformalSocialCare/totalSocialCare
        
        print share_InformalSocialCare
        
        share_UnmetSocialCareNeed = 0
        if totalSocialCareNeed > 0:
            share_UnmetSocialCareNeed = totalUnmetSocialCareNeed/totalSocialCareNeed
        
        
        for house in self.map.occupiedHouses:
            house.totalUnmetSocialCareNeed = sum([x.unmetSocialCareNeed for x in house.occupants])
#        print totalSocialCareNeed
#        if totalInformalSocialCare > 0:
#            print 'Share of in-house informal care: ' + str(float(self.inHouseInformalCare)/totalInformalSocialCare)
#        print share_UnmetSocialCareNeed
#        print ''
        
        outOfWorkSocialCare = [x.outOfWorkSocialCare for x in self.pop.livingPeople]
        totalOWSC = sum(outOfWorkSocialCare)
        shareOWSC = 0
        if totalInformalSocialCare > 0:
            shareOWSC = totalOWSC/totalInformalSocialCare
        totalCostOWSC = sum([x.outOfWorkSocialCare*x.wage for x in self.pop.livingPeople if x.outOfWorkSocialCare > 0])
        
        # By income quintiles
        households = [x for x in self.map.occupiedHouses]
        q1_households = [x for x in households if x.incomeQuintile == 0]
        q1_socialCareNeed = sum([x.totalSocialCareNeed for x in q1_households])
        q2_households = [x for x in households if x.incomeQuintile == 1]
        q2_socialCareNeed = sum([x.totalSocialCareNeed for x in q2_households])
        q3_households = [x for x in households if x.incomeQuintile == 2]
        q3_socialCareNeed = sum([x.totalSocialCareNeed for x in q3_households])
        q4_households = [x for x in households if x.incomeQuintile == 3]
        q4_socialCareNeed = sum([x.totalSocialCareNeed for x in q4_households])
        q5_households = [x for x in households if x.incomeQuintile == 4]
        q5_socialCareNeed = sum([x.totalSocialCareNeed for x in q5_households])
        
        q1_informalSocialCare = sum([x.informalSocialCareReceived for x in q1_households])
        q2_informalSocialCare = sum([x.informalSocialCareReceived for x in q2_households])
        q3_informalSocialCare = sum([x.informalSocialCareReceived for x in q3_households])
        q4_informalSocialCare = sum([x.informalSocialCareReceived for x in q4_households])
        q5_informalSocialCare = sum([x.informalSocialCareReceived for x in q5_households])
        
        q1_outOfWorkSocialCare = sum([x.outOfWorkSocialCare for x in q1_households])
        q2_outOfWorkSocialCare = sum([x.outOfWorkSocialCare for x in q2_households])
        q3_outOfWorkSocialCare = sum([x.outOfWorkSocialCare for x in q3_households])
        q4_outOfWorkSocialCare = sum([x.outOfWorkSocialCare for x in q4_households])
        q5_outOfWorkSocialCare = sum([x.outOfWorkSocialCare for x in q5_households])
        
        q1_formalSocialCare = sum([x.formalSocialCareReceived for x in q1_households])
        q2_formalSocialCare = sum([x.formalSocialCareReceived for x in q2_households])
        q3_formalSocialCare = sum([x.formalSocialCareReceived for x in q3_households])
        q4_formalSocialCare = sum([x.formalSocialCareReceived for x in q4_households])
        q5_formalSocialCare = sum([x.formalSocialCareReceived for x in q5_households])
        
        q1_unmetSocialCareNeed = sum([x.totalUnmetSocialCareNeed for x in q1_households])
        q2_unmetSocialCareNeed = sum([x.totalUnmetSocialCareNeed for x in q2_households])
        q3_unmetSocialCareNeed = sum([x.totalUnmetSocialCareNeed for x in q3_households])
        q4_unmetSocialCareNeed = sum([x.totalUnmetSocialCareNeed for x in q4_households])
        q5_unmetSocialCareNeed = sum([x.totalUnmetSocialCareNeed for x in q5_households])
        
        q1_agents = [x for x in self.pop.livingPeople if x.incomeQuintile == 0]
        q2_agents = [x for x in self.pop.livingPeople if x.incomeQuintile == 1]
        q3_agents = [x for x in self.pop.livingPeople if x.incomeQuintile == 2]
        q4_agents = [x for x in self.pop.livingPeople if x.incomeQuintile == 3]
        q5_agents = [x for x in self.pop.livingPeople if x.incomeQuintile == 4]
        
        unmetSocialCareNeed_Q1 = sum([x.unmetSocialCareNeed for x in q1_agents])
        unmetSocialCareNeed_Q2 = sum([x.unmetSocialCareNeed for x in q2_agents])
        unmetSocialCareNeed_Q3 = sum([x.unmetSocialCareNeed for x in q3_agents])
        unmetSocialCareNeed_Q4 = sum([x.unmetSocialCareNeed for x in q4_agents])
        unmetSocialCareNeed_Q5 = sum([x.unmetSocialCareNeed for x in q5_agents])
        
        mostLeastDeprivedRatio = 0
        if unmetSocialCareNeed_Q5 > 0:
            mostLeastDeprivedRatio = unmetSocialCareNeed_Q1/unmetSocialCareNeed_Q5
            
        totalInformalSocialCare_Q1 = sum([x.informalSocialCareReceived for x in q1_agents])
        totalInformalSocialCare_Q2 = sum([x.informalSocialCareReceived for x in q2_agents])
        totalInformalSocialCare_Q3 = sum([x.informalSocialCareReceived for x in q3_agents])
        totalInformalSocialCare_Q4 = sum([x.informalSocialCareReceived for x in q4_agents])
        totalInformalSocialCare_Q5 = sum([x.informalSocialCareReceived for x in q5_agents])
        
        informalCareRatio = 0
        if totalInformalSocialCare_Q5 > 0:
            informalCareRatio = totalInformalSocialCare_Q1/totalInformalSocialCare_Q5
        
        totalSocialCareNeed_Q1 = sum([x.hoursSocialCareDemand for x in q1_agents])
        totalSocialCareNeed_Q2 = sum([x.hoursSocialCareDemand for x in q2_agents])
        totalSocialCareNeed_Q3 = sum([x.hoursSocialCareDemand for x in q3_agents])
        totalSocialCareNeed_Q4 = sum([x.hoursSocialCareDemand for x in q4_agents])
        totalSocialCareNeed_Q5 = sum([x.hoursSocialCareDemand for x in q5_agents])
        
        careNeedRatio = 0
        if totalSocialCareNeed_Q5 > 0:
            careNeedRatio = totalSocialCareNeed_Q1/totalSocialCareNeed_Q5
            
        hospitalizedQuintilesRatio = 0
        if self.totalHospitalizedByClass[4] > 0:
            hospitalizedQuintilesRatio = float(self.totalHospitalizedByClass[0])/float(self.totalHospitalizedByClass[4])
            
        intubatedQuintilesRatio = 0
        if self.totalIntubatedByClass[4] > 0:
            intubatedQuintilesRatio = float(self.totalIntubatedByClass[0])/float(self.totalIntubatedByClass[4])
        
        deathsQuintilesRatio = 0
        if self.totDeathsByClass[4] > 0:
            deathsQuintilesRatio = float(self.totDeathsByClass[0])/float(self.totDeathsByClass[4])
        
        q1_infected = len([x for x in q1_agents if x.healthStatus == 'infectious'])
        q2_infected = len([x for x in q2_agents if x.healthStatus == 'infectious'])
        q3_infected = len([x for x in q3_agents if x.healthStatus == 'infectious'])
        q4_infected = len([x for x in q4_agents if x.healthStatus == 'infectious'])
        q5_infected = len([x for x in q5_agents if x.healthStatus == 'infectious'])
        
        q1_hospitalized = len([x for x in q1_agents if x.hospitalized == True])
        q2_hospitalized = len([x for x in q2_agents if x.hospitalized == True])
        q3_hospitalized = len([x for x in q3_agents if x.hospitalized == True])
        q4_hospitalized = len([x for x in q4_agents if x.hospitalized == True])
        q5_hospitalized = len([x for x in q5_agents if x.hospitalized == True])
        
        q1_intubated = len([x for x in q1_agents if x.hospitalized == True and (x.symptomsLevel == 'critical' or x.symptomsLevel == 'dead')])
        q2_intubated = len([x for x in q2_agents if x.hospitalized == True and (x.symptomsLevel == 'critical' or x.symptomsLevel == 'dead')])
        q3_intubated = len([x for x in q3_agents if x.hospitalized == True and (x.symptomsLevel == 'critical' or x.symptomsLevel == 'dead')])
        q4_intubated = len([x for x in q4_agents if x.hospitalized == True and (x.symptomsLevel == 'critical' or x.symptomsLevel == 'dead')])
        q5_intubated = len([x for x in q5_agents if x.hospitalized == True and (x.symptomsLevel == 'critical' or x.symptomsLevel == 'dead')])
        
       
        taxPayers = len([x for x in self.pop.livingPeople if x.status == 'student' or x.status == 'worker'])
        self.numTaxpayers.append(taxPayers)
        
        if totalSocialCareNeed == 0:
            familyCareRatio = 0.0
        else:
            familyCareRatio = (totalSocialCareNeed - totalUnmetSocialCareNeed)/totalSocialCareNeed

        ##familyCareRatio = ( totalCareDemandHours - unmetNeed ) / (1.0 * (totalCareDemandHours+0.01))
        self.totalFamilyCare.append(familyCareRatio)
        taxBurden = 0
        if taxPayers > 0:
            taxBurden = ( totalUnmetSocialCareNeed * self.p['hourlyCostOfCare'] * 52.18 ) / ( taxPayers * 1.0 )
        self.totalTaxBurden.append(taxBurden)
        
        ## Count the proportion of adult women who are married
        totalAdultWomen = 0
        totalMarriedAdultWomen = 0

        for person in self.pop.livingPeople:
            age = self.year - person.birthdate
            if person.sex == 'female' and age >= 18:
                totalAdultWomen += 1
                if person.partner != None:
                    totalMarriedAdultWomen += 1
        marriagePropNow = 0
        if float(totalAdultWomen) > 0:
            marriagePropNow = float(totalMarriedAdultWomen)/float(totalAdultWomen)
        self.marriageProp.append(marriagePropNow)
       
        formalChildCare = sum([x.formalChildCareReceived for x in self.pop.livingPeople])
        formalChildCareCost = formalChildCare*self.p['priceChildCare']
        householdsIncome = sum([x.householdIncome for x in self.map.occupiedHouses])
        childcareIncomeShare = 0
        if householdsIncome > 0:
            childcareIncomeShare = formalChildCareCost/householdsIncome
      
        children = [x for x in self.pop.livingPeople if x.age < self.p['ageTeenagers']]
        totalChildCareNeed = sum([x.netChildCareDemand for x in children])
        unmetChildCareNeed = sum([x.unmetChildCareNeed for x in children])
        
#        print 'Total child care need: ' + str(totalChildCareNeed)
#        print 'Unmet child care need: ' + str(unmetChildCareNeed)
        
        totalInformalChildCare = sum([x.informalChildCareReceived for x in children])
        shareInformalChildCare = 0
        if totalChildCareNeed > 0:
            shareInformalChildCare = totalInformalChildCare/totalChildCareNeed
       
        # Social care stats
        over16Pop = [x for x in self.pop.livingPeople if x.age > 16]
        totalSuppliers = [x for x in over16Pop if x.socialWork > 0]
        shareCareGivers = 0
        if float(len(over16Pop)) > 0:
            shareCareGivers = float(len(totalSuppliers))/float(len(over16Pop))
        familyCarers = [x for x in over16Pop if x.careForFamily == True]
        
        malesOver16 = [x for x in over16Pop if x.sex == 'male']
        femalesOver16 = [x for x in over16Pop if x.sex == 'female']
        maleSuppliers = [x for x in malesOver16 if x.socialWork > 0]
        femaleSuppliers = [x for x in femalesOver16 if x.socialWork > 0]
        ratioFemaleMaleCarers = 0
        if len(totalSuppliers) > 0:
            ratioFemaleMaleCarers = float(len(femaleSuppliers))/float(len(totalSuppliers))
        shareMaleCarers = 0
        if len(malesOver16) > 0:
            shareMaleCarers = float(len(maleSuppliers))/float(len(malesOver16))
        shareFemaleCarers = 0
        if len(femalesOver16) > 0:
            shareFemaleCarers = float(len(femaleSuppliers))/float(len(femalesOver16))
        
        workers = [x for x in self.pop.livingPeople if x.status == 'worker' and x.careNeedLevel < 3]
        employedMales = [x for x in workers if x.sex == 'male']
        employedFemales = [x for x in workers if x.sex == 'female']
        
        meanMaleWage = np.mean([x.wage for x in employedMales])
        meanFemaleWage = np.mean([x.wage for x in employedFemales])
        ratioWage = 0
        if meanMaleWage > 0:
            ratioWage = meanFemaleWage/meanMaleWage
        
        meanMaleIncome = np.mean([x.income for x in employedMales])
        meanFemaleIncome = np.mean([x.income for x in employedFemales])
        ratioIncome = 0
        if meanMaleIncome > 0:
            ratioIncome = meanFemaleIncome/meanMaleIncome
        
        informalSocialCarers = [x for x in self.pop.livingPeople if x.socialWork > 0]
        informalSocialReceivers = [x for x in self.pop.livingPeople if x.informalSocialCareReceived > 0]
        informalCaresSupplied = [x.socialWork for x in informalSocialCarers]
        shareFamilyCarer = 0
        if len(informalSocialCarers) > 0:
            shareFamilyCarer = float(len(familyCarers))/float(len(informalSocialCarers))
        
        over20Hours_FamilyCarers = [x for x in familyCarers if x.socialWork > 20]
        share_over20Hours_FamilyCarers = 0
        if len(familyCarers) > 0:
            share_over20Hours_FamilyCarers = float(len(over20Hours_FamilyCarers))/float(len(familyCarers))
        averageHoursOfCare = np.mean(informalCaresSupplied)
        carers_40to64 = [x for x in informalSocialCarers if x.age >= 40 and x.age <= 64]
        over65_carers = [x for x in informalSocialCarers if x.age >= 65]
        share_40to64_carers = 0
        if len(informalSocialCarers) > 0:
            share_40to64_carers = float(len(carers_40to64))/float(len(informalSocialCarers))
        share_over65_carers = 0
        if len(informalSocialCarers) > 0:
            share_over65_carers = float(len(over65_carers))/float(len(informalSocialCarers))
            
        over70_carers = [x for x in informalSocialCarers if x.age >= 70]  
        TenPlusHours_over70 = [x for x in over70_carers if x.age >= 70 if x.socialWork > 10]
        share_10PlusHours_over70 = 0
        if len(over70_carers) > 0:
            share_10PlusHours_over70 = float(len(TenPlusHours_over70))/float(len(over70_carers))
            
        
        self.costTaxFreeSocialCare = totalFormalSocialCare*self.p['priceSocialCare']*self.p['socialCareTaxFreeRate']
        
        publicCareToGDP = 0
        if self.grossDomesticProduct > 0:
            publicCareToGDP = self.costPublicSocialCare/self.grossDomesticProduct
        
        isolationRates = [x.isolationRate for x in self.pop.livingPeople]
        self.medianIsolationRate = np.median(isolationRates)
        self.meanIsolationRate = np.mean(isolationRates)
        
        
        self.susceptibles = len([x for x in self.pop.livingPeople if x.healthStatus == 'susceptible'])
        print self.susceptibles
        self.exposed = len([x for x in self.pop.livingPeople if x.healthStatus == 'exposed'])
        self.infectious = len([x for x in self.pop.livingPeople if x.healthStatus == 'infectious'])
        self.recovered = len([x for x in self.pop.livingPeople if x.healthStatus == 'recovered'])
        self.asymptomatic = len([x for x in self.pop.livingPeople if x.symptomatic == False and x.healthStatus != 'recovered'])
        self.mildSymptomatic = len([x for x in self.pop.livingPeople if x.symptomatic == True and x.symptomsLevel == 'mild'])
        self.hospitalized = len([x for x in self.pop.livingPeople if x.hospitalized == True])
        self.over70Hospitalized = len([x for x in self.pop.livingPeople if x.hospitalized == True and x.age >= 70])
        self.intubated = len([x for x in self.pop.livingPeople if x.hospitalized == True and (x.symptomsLevel == 'critical' or x.symptomsLevel == 'dead')])
        self.over70Intubated = len([x for x in self.pop.livingPeople if x.hospitalized == True and (x.symptomsLevel == 'critical' or x.symptomsLevel == 'dead') and x.age >= 70])
  
        experiencedEvents = [x.experiencedEvent for x in self.map.allHouses if x.id in self.householdIndexes]
        kindOfEvents = [x.kindOfEvent for x in self.map.allHouses if x.id in self.householdIndexes]
        infectionEvents = [x.infectionEvent for x in self.map.allHouses if x.id in self.householdIndexes]
        betas = [x.beta for x in self.map.allHouses if x.id in self.householdIndexes]
        contactReductionRates = [x.contactReductionRate for x in self.map.allHouses if x.id in self.householdIndexes]
        isolationFactors = [x.isolationFactor for x in self.map.allHouses if x.id in self.householdIndexes]
        individualIsolations = [x.individualIsolation for x in self.map.allHouses if x.id in self.householdIndexes]
        neighborIsolations = [x.neighborIsolation for x in self.map.allHouses if x.id in self.householdIndexes]  
        
        print 'No more occupied houses: ' + str(self.housesNotMoreOccupied)
        
        # self.gdpRatio = float(sum([x.workingShare for x in self.pop.livingPeople]))/float(len(self.pop.livingPeople))
        housesOutputs = [experiencedEvents[0], kindOfEvents[0], infectionEvents[0], betas[0], contactReductionRates[0], 
                         isolationFactors[0], individualIsolations[0], neighborIsolations[0], experiencedEvents[1], kindOfEvents[1], 
                         infectionEvents[1], betas[1], contactReductionRates[1], isolationFactors[1], individualIsolations[1], 
                         neighborIsolations[1], experiencedEvents[2], kindOfEvents[2], infectionEvents[2], betas[2], 
                         contactReductionRates[2], isolationFactors[2], individualIsolations[2], neighborIsolations[2],
                         experiencedEvents[3], kindOfEvents[3], infectionEvents[3], betas[3], contactReductionRates[3], 
                         isolationFactors[3], individualIsolations[3], neighborIsolations[3],
                         experiencedEvents[4], kindOfEvents[4], infectionEvents[4], betas[4], contactReductionRates[4], 
                         isolationFactors[4], individualIsolations[4], neighborIsolations[4], experiencedEvents[5], kindOfEvents[5], 
                         infectionEvents[5], betas[5], contactReductionRates[5], isolationFactors[5], individualIsolations[5], 
                         neighborIsolations[5], experiencedEvents[6], kindOfEvents[6], infectionEvents[6], betas[6], 
                         contactReductionRates[6], isolationFactors[6], individualIsolations[6], neighborIsolations[6],
                         experiencedEvents[7], kindOfEvents[7], infectionEvents[7], betas[7], contactReductionRates[7], 
                         isolationFactors[7], individualIsolations[7], neighborIsolations[7],
                         experiencedEvents[8], kindOfEvents[8], infectionEvents[8], betas[8], contactReductionRates[8], 
                         isolationFactors[8], individualIsolations[8], neighborIsolations[8], experiencedEvents[9], kindOfEvents[9], 
                         infectionEvents[9], betas[9], contactReductionRates[9], isolationFactors[9], individualIsolations[9], 
                         neighborIsolations[9], experiencedEvents[10], kindOfEvents[10], infectionEvents[10], betas[10], 
                         contactReductionRates[10], isolationFactors[10], individualIsolations[10], neighborIsolations[10],
                         experiencedEvents[11], kindOfEvents[11], infectionEvents[11], betas[11], contactReductionRates[11], 
                         isolationFactors[11], individualIsolations[11], neighborIsolations[11],
                         experiencedEvents[12], kindOfEvents[12], infectionEvents[12], betas[12], contactReductionRates[12], 
                         isolationFactors[12], individualIsolations[12], neighborIsolations[12], experiencedEvents[13], kindOfEvents[13], 
                         infectionEvents[13], betas[13], contactReductionRates[13], isolationFactors[13], individualIsolations[13], 
                         neighborIsolations[13], experiencedEvents[14], kindOfEvents[14], infectionEvents[14], betas[14], 
                         contactReductionRates[14], isolationFactors[14], individualIsolations[14], neighborIsolations[14],
                         experiencedEvents[15], kindOfEvents[15], infectionEvents[15], betas[15], contactReductionRates[15], 
                         isolationFactors[15], individualIsolations[15], neighborIsolations[15],
                         experiencedEvents[16], kindOfEvents[16], infectionEvents[16], betas[16], contactReductionRates[16], 
                         isolationFactors[16], individualIsolations[16], neighborIsolations[16], experiencedEvents[17], kindOfEvents[17], 
                         infectionEvents[17], betas[17], contactReductionRates[17], isolationFactors[17], individualIsolations[17], 
                         neighborIsolations[17], experiencedEvents[18], kindOfEvents[18], infectionEvents[18], betas[18], 
                         contactReductionRates[18], isolationFactors[18], individualIsolations[18], neighborIsolations[18],
                         experiencedEvents[19], kindOfEvents[19], infectionEvents[19], betas[19], contactReductionRates[19], 
                         isolationFactors[19], individualIsolations[19], neighborIsolations[19]]
        
        
        outputs = [day, self.susceptibles, self.exposed, self.infectious, self.recovered, 
                   self.totalDeaths, self.hospitalized, self.intubated, self.deathsForCovid, self.asymptomatic, self.mildSymptomatic,
                   self.newCases, self.newExposed, self.over70Hospitalized, self.over70Intubated, self.lockdown, q1_infected, q1_hospitalized, q1_intubated,
                   q2_infected, q2_hospitalized, q2_intubated, q3_infected, q3_hospitalized, q3_intubated, q4_infected, q4_hospitalized, q4_intubated,
                   q5_infected, q5_hospitalized, q5_intubated, self.totalHospitalized, self.cumulatedHospitalizations, self.cumulatedICUs,
                   self.hospitalPopulation,self.icuPopulation,
                   self.infectedByClass[0], self.infectedByClass[1], self.infectedByClass[2], self.infectedByClass[3], self.infectedByClass[4],
                   self.infectedByAge[0], self.infectedByAge[1], self.infectedByAge[2], self.infectedByAge[3], self.infectedByAge[4],
                   self.infectedByAge[5], self.infectedByAge[6], self.infectedByAge[7], self.infectedByAge[8],
                   self.hospitalizedByClass[0], self.hospitalizedByClass[1], self.hospitalizedByClass[2], self.hospitalizedByClass[3], self.hospitalizedByClass[4],
                   hospitalizedQuintilesRatio, self.hospitalizedByAge[0], self.hospitalizedByAge[1], self.hospitalizedByAge[2], self.hospitalizedByAge[3], 
                   self.hospitalizedByAge[4], self.hospitalizedByAge[5], self.hospitalizedByAge[6], self.hospitalizedByAge[7], self.hospitalizedByAge[8],
                   self.intubatedByClass[0], self.intubatedByClass[1], self.intubatedByClass[2], self.intubatedByClass[3], self.intubatedByClass[4],
                   intubatedQuintilesRatio, self.intubatedByAge[0], self.intubatedByAge[1], self.intubatedByAge[2], self.intubatedByAge[3], self.intubatedByAge[4],
                   self.intubatedByAge[5], self.intubatedByAge[6], self.intubatedByAge[7], self.intubatedByAge[8],
                   self.symptomaticByClass[0], self.symptomaticByClass[1], self.symptomaticByClass[2], self.symptomaticByClass[3], self.symptomaticByClass[4],
                   self.symptomaticByAge[0], self.symptomaticByAge[1], self.symptomaticByAge[2], self.symptomaticByAge[3], self.symptomaticByAge[4],
                   self.symptomaticByAge[5], self.symptomaticByAge[6], self.symptomaticByAge[7], self.symptomaticByAge[8],
                   self.deathsByClass[0], self.deathsByClass[1], self.deathsByClass[2], self.deathsByClass[3], self.deathsByClass[4], deathsQuintilesRatio,
                   self.deathsByAge[0], self.deathsByAge[1], self.deathsByAge[2], self.deathsByAge[3], self.deathsByAge[4],
                   self.deathsByAge[5], self.deathsByAge[6], self.deathsByAge[7], self.deathsByAge[8], 
                   unmetSocialCareNeed_Q1, unmetSocialCareNeed_Q2, unmetSocialCareNeed_Q3, unmetSocialCareNeed_Q4, unmetSocialCareNeed_Q5, mostLeastDeprivedRatio,
                   totalInformalSocialCare_Q1, totalInformalSocialCare_Q2, totalInformalSocialCare_Q3, totalInformalSocialCare_Q4, totalInformalSocialCare_Q5,
                   totalSocialCareNeed_Q1, totalSocialCareNeed_Q2, totalSocialCareNeed_Q3, totalSocialCareNeed_Q4, totalSocialCareNeed_Q5,
                   informalCareRatio, careNeedRatio, self.medianIsolationRate, self.meanIsolationRate, self.mobilityReduction_Q1,
                   self.mobilityReduction_Q2, self.mobilityReduction_Q3, self.mobilityReduction_Q4, self.mobilityReduction_Q5, self.mobilityReduction_A1,
                   self.mobilityReduction_A2, self.mobilityReduction_A3, self.mobilityReduction_A4, self.mobilityReduction_A5, self.mobilityReduction_A6,
                   self.mobilityReduction_A7, self.mobilityReduction_A8, self.mobilityReduction_A9, self.totalAttendants, self.cumulatedAbsences, self.periodTests]
#                   self.shareHospitalized1To11, self.shareHospitalized12To18, self.shareHospitalized19To39, self.shareHospitalized40To64,
#                   self.shareHospitalizedOver65, self.shareIntubated1To11, self.shareIntubated12To18, self.shareIntubated19To39,
#                   self.shareIntubated40To64, self.shareIntubatedOver65, self.shareDeaths1To11, self.shareDeaths12To18, self.shareDeaths19To39,
#                   self.shareDeaths40To64, self.shareDeathsOver65, self.Age1To11Share_H, self.Age12To18Share_H, self.Age19To39Share_H,
#                   self.Age40To64Share_H, self.Over65Share_H, self.Age1To11Share_I, self.Age12To18Share_I, self.Age19To39Share_I,
#                   self.Age40To64Share_I, self.Over65Share_I]
        
        dataMapFile = 'DataMap_' + str(self.year) + '.csv'
        if not os.path.exists(dataMapFolder):
            os.makedirs(dataMapFolder)
        with open(os.path.join(dataMapFolder, dataMapFile), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.dataMap))
            for house in self.map.allHouses:
                data = [house.town.x, house.town.y, house.x, house.y, len(house.occupants), house.totalUnmetSocialCareNeed]
                writer.writerow(data)
        
                
        householdFile = 'DataHousehold_' + str(self.year) + '.csv'
        if not os.path.exists(dataHouseholdFolder):
            os.makedirs(dataHouseholdFolder)
        with open(os.path.join(dataHouseholdFolder, householdFile), "w") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow((self.householdData))
            for member in self.displayHouse.occupants:
                data = [member.id, member.sex, member.age, member.careNeedLevel]
                writer.writerow(data)    
        
        houseData = [self.year, self.displayHouse.name, len(self.displayHouse.occupants)]
        with open(os.path.join(policyFolder, "HouseData.csv"), "a") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow(houseData)
        
        malesByNeed = []
        for i in range(int(self.p['numCareLevels'])):
            malesByAge = []
            for j in range(25):
                subgroup = [x for x in self.pop.livingPeople if x.sex == 'male' and int(x.age/5) == j and x.careNeedLevel == i]
                malesByAge.append(len(subgroup))
            malesByNeed.append(malesByAge)
            
        femalesByNeed = []
        for i in range(int(self.p['numCareLevels'])):
            femalesByAge = []
            for j in range(25):
                subgroup = [x for x in self.pop.livingPeople if x.sex == 'female' and int(x.age/5) == j and x.careNeedLevel == i]
                femalesByAge.append(len(subgroup))
            femalesByNeed.append(femalesByAge)
        
        for i in range(int(self.p['numCareLevels'])):
            pyramidData = [self.year]
            pyramidData.extend(malesByNeed[i])
            fileName = 'Pyramid_Male_' + str(i) + '.csv'
            with open(os.path.join(policyFolder, fileName), "a") as file:
                writer = csv.writer(file, delimiter = ",", lineterminator='\r')
                writer.writerow(pyramidData)
            
        for i in range(int(self.p['numCareLevels'])):
            pyramidData = [self.year]
            pyramidData.extend(femalesByNeed[i])
            fileName = 'Pyramid_Female_' + str(i) + '.csv'
            with open(os.path.join(policyFolder, fileName), "a") as file:
                writer = csv.writer(file, delimiter = ",", lineterminator='\r')
                writer.writerow(pyramidData)
            
        houseData = [self.year, self.displayHouse.name, len(self.displayHouse.occupants)]
        with open(os.path.join(policyFolder, "HouseData.csv"), "a") as file:
            writer = csv.writer(file, delimiter = ",", lineterminator='\r')
            writer.writerow(houseData)
            
        if day == 0:
            if not os.path.exists(policyFolder):
                os.makedirs(policyFolder)
            with open(os.path.join(policyFolder, "Outputs.csv"), "w") as file:
                writer = csv.writer(file, delimiter = ",", lineterminator='\r')
                writer.writerow((self.Outputs))
                writer.writerow(outputs)
        else:
            with open(os.path.join(policyFolder, "Outputs.csv"), "a") as file:
                writer = csv.writer(file, delimiter = ",", lineterminator='\r')
                writer.writerow(outputs)
                
        if day == 0:
            if not os.path.exists(policyFolder):
                os.makedirs(policyFolder)
            with open(os.path.join(policyFolder, "householdsOutputs.csv"), "w") as file:
                writer = csv.writer(file, delimiter = ",", lineterminator='\r')
                writer.writerow((self.housesOutputs))
                writer.writerow(housesOutputs)
        else:
            with open(os.path.join(policyFolder, "householdsOutputs.csv"), "a") as file:
                writer = csv.writer(file, delimiter = ",", lineterminator='\r')
                writer.writerow(housesOutputs)
        
        self.marriageTally = 0
        self.divorceTally = 0
        
        ## Some extra debugging stuff just to check that all
        ## the lists are behaving themselves
        if self.p['verboseDebugging']:
            peopleCount = 0
            for i in self.pop.allPeople:
                if i.dead == False:
                    peopleCount += 1
            print "True pop counting non-dead people in allPeople list = ", peopleCount

            peopleCount = 0
            for h in self.map.occupiedHouses:
                peopleCount += len(h.occupants)
            print "True pop counting occupants of all occupied houses = ", peopleCount

            peopleCount = 0
            for h in self.map.allHouses:
                peopleCount += len(h.occupants)
            print "True pop counting occupants of ALL houses = ", peopleCount

            tally = [ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
            for h in self.map.occupiedHouses:
                tally[len(h.occupants)] += 1
            for i in range(len(tally)):
                if tally[i] > 0:
                    print i, tally[i]
            print

            
    def healthCareCost(self):
        
        self.totalHospitalizationCost = self.p['costHospitalizationPerDay']*len([x for x in self.pop.allPeople if x.hospitalized == True])
#        peopleWithUnmetNeed = [x for x in self.pop.livingPeople if x.careNeedLevel > 0]
#        for person in peopleWithUnmetNeed:
#            needLevelFactor = math.pow(self.p['needLevelParam'], person.careNeedLevel)
#            unmetSocialCareFactor = math.pow(self.p['unmetSocialCareParam'], person.averageShareUnmetNeed)
#            averageHospitalization = self.p['hospitalizationParam']*needLevelFactor*unmetSocialCareFactor
#            self.totalHospitalizationCost += averageHospitalization*self.p['costHospitalizationPerDay']
        self.hospitalizationCost.append(self.totalHospitalizationCost)
        
    def initializeCanvas(self):
        """Put up a TKInter canvas window to animate the simulation."""
        self.canvas.pack()

        ## Draw some numbers for the population pyramid that won't be redrawn each time
        for a in range(0,int(self.p['num5YearAgeClasses'])):
            self.canvas.create_text(170, 385 - (10 * a),
                                    text=str(5*a) + '-' + str(5*a+4),
                                    font='Helvetica 6',
                                    fill='white')

        ## Draw the overall map, including towns and houses (occupied houses only)
        for t in self.map.towns:
            xBasic = 580 + (t.x * self.p['pixelsPerTown'])
            yBasic = 15 + (t.y * self.p['pixelsPerTown'])
            self.canvas.create_rectangle(xBasic, yBasic,
                                         xBasic+self.p['pixelsPerTown'],
                                         yBasic+self.p['pixelsPerTown'],
                                         outline='grey',
                                         state = 'hidden' )

        for h in self.map.allHouses:
            t = h.town
            xBasic = 580 + (t.x * self.p['pixelsPerTown'])
            yBasic = 15 + (t.y * self.p['pixelsPerTown'])
            xOffset = xBasic + 2 + (h.x * 2)
            yOffset = yBasic + 2 + (h.y * 2)

            outlineColour = fillColour = self.p['houseSizeColour'][h.size]
            width = 1

            h.icon = self.canvas.create_rectangle(xOffset,yOffset,
                                                  xOffset + width, yOffset + width,
                                                  outline=outlineColour,
                                                  fill=fillColour,
                                                  state = 'normal' )

        self.canvas.update()
        time.sleep(0.5)
        self.canvas.update()

        for h in self.map.allHouses:
            self.canvas.itemconfig(h.icon, state='hidden')

        for h in self.map.occupiedHouses:
            self.canvas.itemconfig(h.icon, state='normal')

        self.canvas.update()
        self.updateCanvas()

    def updateCanvas(self):
        """Update the appearance of the graphics canvas."""

        ## First we clean the canvas off; some items are redrawn every time and others are not
        self.canvas.delete('redraw')

        ## Now post the current year and the current population size
        self.canvas.create_text(self.p['dateX'],
                                self.p['dateY'],
                                text='Year: ' + str(self.year),
                                font = self.p['mainFont'],
                                fill = self.p['fontColour'],
                                tags = 'redraw')
        self.canvas.create_text(self.p['popX'],
                                self.p['popY'],
                                text='Pop: ' + str(len(self.pop.livingPeople)),
                                font = self.p['mainFont'],
                                fill = self.p['fontColour'],
                                tags = 'redraw')

        self.canvas.create_text(self.p['popX'],
                                self.p['popY'] + 30,
                                text='Ever: ' + str(len(self.pop.allPeople)),
                                font = self.p['mainFont'],
                                fill = self.p['fontColour'],
                                tags = 'redraw')

        ## Also some other stats, but not on the first display
        if self.year > self.p['startYear']:
            self.canvas.create_text(350,20,
                                    text='Avg household: ' + str ( round ( self.avgHouseholdSize[-1] , 2 ) ),
                                    font = 'Helvetica 11',
                                    fill = 'white',
                                    tags = 'redraw')
            self.canvas.create_text(350,40,
                                    text='Marriages: ' + str(self.numMarriages[-1]),
                                    font = 'Helvetica 11',
                                    fill = 'white',
                                    tags = 'redraw')
            self.canvas.create_text(350,60,
                                    text='Divorces: ' + str(self.numDivorces[-1]),
                                    font = 'Helvetica 11',
                                    fill = 'white',
                                    tags = 'redraw')
            self.canvas.create_text(350,100,
                                    text='Total care demand: ' + str(round(self.totalCareDemand[-1], 0 ) ),
                                    font = 'Helvetica 11',
                                    fill = 'white',
                                    tags = 'redraw')
            self.canvas.create_text(350,120,
                                    text='Num taxpayers: ' + str(round(self.numTaxpayers[-1], 0 ) ),
                                    font = 'Helvetica 11',
                                    fill = 'white',
                                    tags = 'redraw')
            self.canvas.create_text(350,140,
                                    text='Family care ratio: ' + str(round(100.0 * self.totalFamilyCare[-1], 0 ) ) + "%",
                                    font = 'Helvetica 11',
                                    fill = 'white',
                                    tags = 'redraw')
            self.canvas.create_text(350,160,
                                    text='Tax burden: ' + str(round(self.totalTaxBurden[-1], 0 ) ),
                                    font = 'Helvetica 11',
                                    fill = 'white',
                                    tags = 'redraw')
            self.canvas.create_text(350,180,
                                    text='Marriage prop: ' + str(round(100.0 * self.marriageProp[-1], 0 ) ) + "%",
                                    font = 'Helvetica 11',
                                    fill = self.p['fontColour'],
                                    tags = 'redraw')

        

        ## Draw the population pyramid split by care categories
        for a in range(0,int(self.p['num5YearAgeClasses'])):
            malePixel = 153
            femalePixel = 187
            for c in range(0,int(self.p['numCareLevels'])):
                mWidth = self.pyramid.maleData[a,c]
                fWidth = self.pyramid.femaleData[a,c]

                if mWidth > 0:
                    self.canvas.create_rectangle(malePixel, 380 - (10*a),
                                                 malePixel - mWidth, 380 - (10*a) + 9,
                                                 outline=self.p['careLevelColour'][c],
                                                 fill=self.p['careLevelColour'][c],
                                                 tags = 'redraw')
                malePixel -= mWidth
                
                if fWidth > 0:
                    self.canvas.create_rectangle(femalePixel, 380 - (10*a),
                                                 femalePixel + fWidth, 380 - (10*a) + 9,
                                                 outline=self.p['careLevelColour'][c],
                                                 fill=self.p['careLevelColour'][c],
                                                 tags = 'redraw')
                femalePixel += fWidth

        ## Draw in the display house and the people who live in it
        if len(self.displayHouse.occupants) < 1:
            ## Nobody lives in the display house any more, choose another
            if self.nextDisplayHouse != None:
                self.displayHouse = self.nextDisplayHouse
                self.nextDisplayHouse = None
            else:
                self.displayHouse = random.choice(self.pop.livingPeople).house
                self.textUpdateList.append(str(self.year) + ": Display house empty, going to " + self.displayHouse.name + ".")
                messageString = "Residents: "
                for k in self.displayHouse.occupants:
                    messageString += "#" + str(k.id) + " "
                self.textUpdateList.append(messageString)
            

        outlineColour = self.p['houseSizeColour'][self.displayHouse.size]
        self.canvas.create_rectangle( 50, 450, 300, 650,
                                      outline = outlineColour,
                                      tags = 'redraw' )
        self.canvas.create_text ( 60, 660,
                                  text="Display house " + self.displayHouse.name,
                                  font='Helvetica 10',
                                  fill='white',
                                  anchor='nw',
                                  tags='redraw')
                                  

        ageBracketCounter = [ 0, 0, 0, 0, 0 ]

        for i in self.displayHouse.occupants:
            age = self.year - i.birthdate
            ageBracket = age / 20
            if ageBracket > 4:
                ageBracket = 4
            careClass = i.careNeedLevel
            sex = i.sex
            idNumber = i.id
            self.drawPerson(age,ageBracket,ageBracketCounter[ageBracket],careClass,sex,idNumber)
            ageBracketCounter[ageBracket] += 1


        ## Draw in some text status updates on the right side of the map
        ## These need to scroll up the screen as time passes

        if len(self.textUpdateList) > self.p['maxTextUpdateList']:
            excess = len(self.textUpdateList) - self.p['maxTextUpdateList']
            self.textUpdateList = self.textUpdateList[excess:excess+self.p['maxTextUpdateList']]

        baseX = 1035
        baseY = 30
        for i in self.textUpdateList:
            self.canvas.create_text(baseX,baseY,
                                    text=i,
                                    anchor='nw',
                                    font='Helvetica 9',
                                    fill = 'white',
                                    width = 265,
                                    tags = 'redraw')
            baseY += 30

        ## Finish by updating the canvas and sleeping briefly in order to allow people to see it
        self.canvas.update()
        if self.p['delayTime'] > 0.0:
            time.sleep(self.p['delayTime'])


    def drawPerson(self, age, ageBracket, counter, careClass, sex, idNumber):
        baseX = 70 + ( counter * 30 )
        baseY = 620 - ( ageBracket * 30 )

        fillColour = self.p['careLevelColour'][careClass]

        self.canvas.create_oval(baseX,baseY,baseX+6,baseY+6,
                                fill=fillColour,
                                outline=fillColour,tags='redraw')
        if sex == 'male':
            self.canvas.create_rectangle(baseX-2,baseY+6,baseX+8,baseY+12,
                                fill=fillColour,outline=fillColour,tags='redraw')
        else:
            self.canvas.create_polygon(baseX+2,baseY+6,baseX-2,baseY+12,baseX+8,baseY+12,baseX+4,baseY+6,
                                fill=fillColour,outline=fillColour,tags='redraw')
        self.canvas.create_rectangle(baseX+1,baseY+13,baseX+5,baseY+20,
                                     fill=fillColour,outline=fillColour,tags='redraw')
            
            
            
        self.canvas.create_text(baseX+11,baseY,
                                text=str(age),
                                font='Helvetica 6',
                                fill='white',
                                anchor='nw',
                                tags='redraw')
        self.canvas.create_text(baseX+11,baseY+8,
                                text=str(idNumber),
                                font='Helvetica 6',
                                fill='grey',
                                anchor='nw',
                                tags='redraw')


    def doGraphs(self):
        """Plot the graphs needed at the end of one run."""
        
        

        p1, = pylab.plot(self.times,self.pops,color="red")
        p2, = pylab.plot(self.times,self.numTaxpayers,color="blue")
        pylab.legend([p1, p2], ['Total population', 'Taxpayers'],loc='lower right')
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Number of people')
        pylab.xlabel('Year')
        pylab.savefig('popGrowth.pdf')
        pylab.show()
        pylab.close()

        pylab.plot(self.times,self.avgHouseholdSize,color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Average household size')
        pylab.xlabel('Year')
        pylab.savefig('avgHousehold.pdf')
        pylab.show()
        pylab.close()

        p1, = pylab.plot(self.times,self.totalCareDemand,color="red")
        p2, = pylab.plot(self.times,self.totalCareSupply,color="blue")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.legend([p1, p2], ['Care demand', 'Total theoretical supply'],loc='lower right')
        pylab.ylabel('Total hours per week')
        pylab.xlabel('Year')
        pylab.savefig('totalCareSituation.pdf')
        pylab.show()

        pylab.plot(self.times,self.totalFamilyCare,color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Proportion of informal social care')
        pylab.xlabel('Year')
        pylab.savefig('informalCare.pdf')
        pylab.show()

        pylab.plot(self.times,self.totalTaxBurden,color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Care costs in pounds per taxpayer per year')
        pylab.xlabel('Year')
        pylab.savefig('taxBurden.pdf')
        pylab.show()

        pylab.plot(self.times,self.marriageProp,color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Proportion of married adult women')
        pylab.xlabel('Year')
        pylab.savefig('marriageProp.pdf')
        pylab.savefig('marriageProp.png')
        pylab.show()
        
        pylab.plot(self.times,self.shareLoneParents,color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Share of Lone Parents')
        pylab.xlabel('Year')
        pylab.savefig('shareLoneParents.pdf')
        pylab.savefig('shareLoneParents.png')
        pylab.show()
        
        pylab.plot(self.times, self.shareUnmetNeed, color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Share of Unmet Care Need')
        pylab.xlabel('Year')
        pylab.savefig('shareUnmetCareNeed.pdf')
        pylab.savefig('shareUnmetCareNeed.png')
        pylab.show()
        
        pylab.plot(self.times, self.hospitalizationCost, color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Hospitalisation Cost')
        pylab.xlabel('Year')
        pylab.savefig('hospitalisationCost.pdf')
        pylab.savefig('hospitalisationCost.png')
        pylab.show()
        
        pylab.plot(self.times, self.employmentRate, color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Employment Rate')
        pylab.xlabel('Year')
        pylab.savefig('employmentRate.pdf')
        pylab.savefig('employmentRate.png')
        pylab.show()
        
        pylab.plot(self.times, self.publicCareProvision, color="red")
        pylab.xlim(xmin=self.p['statsCollectFrom'])
        pylab.ylabel('Public Care Provision')
        pylab.xlabel('Year')
        pylab.savefig('publicCareProvision.pdf')
        pylab.savefig('publicCareProvision.png')
        pylab.show()
        
        y_pos = np.arange(len(self.sesPopsShares))
        plt.bar(y_pos, self.sesPopsShares)
        plt.ylabel('SES Populations')
        plt.show()
        
        lenFrequency = len(self.incomeDistribution)
        individualIncomeFrequencies = [0]*lenFrequency

        dK = np.random.normal(0, self.p['wageVar'])
        indDist = np.random.choice(self.incomesPercentiles, len(self.individualIncomes))*math.exp(dK)
        for i in indDist:
            ind = int(i/1000)
            if ind > -1 and ind < lenFrequency:
                individualIncomeFrequencies[ind] += 1
                
        y_pos = np.arange(lenFrequency)
        plt.bar(y_pos, individualIncomeFrequencies)
        plt.ylabel('individual frequency (empirical)')
        plt.show()
        
        lenFrequency = len(self.incomeDistribution)
        individualIncomeFrequencies = [0]*lenFrequency
        for i in self.individualIncomes:
            ind = int(i/1000)
            if ind > -1 and ind < lenFrequency:
                individualIncomeFrequencies[ind] += 1
                
        y_pos = np.arange(lenFrequency)
        plt.bar(y_pos, individualIncomeFrequencies)
        plt.ylabel('individual frequency (simulated)')
        plt.show()
        
        df = pd.DataFrame()
        df[0] = self.individualIncomes
        df[1] = indDist
        fig, ax = plt.subplots(1,1)
        for s in df.columns:
            df[s].plot(kind='density')
        fig.show()
    

class PopPyramid:
    """Builds a data object for storing population pyramid data in."""
    def __init__ (self, ageClasses, careLevels):
        self.maleData = pylab.zeros((int(ageClasses), int(careLevels)),dtype=int)
        self.femaleData = pylab.zeros((int(ageClasses), int(careLevels)),dtype=int)

    def update(self, year, ageClasses, careLevels, pixelFactor, people):
        ## zero the two arrays
        for a in range (int(ageClasses)):
            for c in range (int(careLevels)):
                self.maleData[a,c] = 0
                self.femaleData[a,c] = 0
        ## tally up who belongs in which category
        for i in people:
            ageClass = ( year - i.birthdate ) / 5
            if ageClass > ageClasses - 1:
                ageClass = ageClasses - 1
            careClass = i.careNeedLevel
            if i.sex == 'male':
                self.maleData[int(ageClass), int(careClass)] += 1
            else:
                self.femaleData[int(ageClass), int(careClass)] += 1

        ## normalize the totals into pixels
        total = len(people)        
        for a in range (int(ageClasses)):
            for c in range (int(careLevels)):
                self.maleData[a,c] = pixelFactor * self.maleData[a,c] / total
                self.femaleData[a,c] = pixelFactor * self.femaleData[a,c] / total

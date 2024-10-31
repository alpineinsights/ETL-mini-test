
import streamlit as st
import anthropic
import voyageai
from llama_parse import LlamaParse
from qdrant_client import QdrantClient
from llama_index.embeddings.voyageai import VoyageEmbedding
import tempfile
import shutil
from datetime import datetime
import requests
import xml.etree.ElementTree as ET
from urllib.parse import unquote
import json
from pathlib import Path
import os
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

st.set_page_config(page_title="PDF Processing Pipeline", page_icon="📚", layout="wide")

STATE_FILE = "./.processed_urls.json"

# Default context prompt
DEFAULT_PROMPT = """Give a short succinct context to situate this chunk within the overall enclosed document boader context for the purpose of improving similarity search retrieval of the chunk. 
Make sure to list:
1. The name of the main company mentioned AND any other secondary companies mentioned if applicable. ONLY use company names exact spellings from the list below to facilitate similarity search retrieval.
2. The apparent date of the document (YYYY.MM.DD)
3. Any fiscal period mentioned. ALWAYS use BOTH abreviated tags (e.g. Q1 2024, Q2 2024, H1 2024) AND more verbose tags (e.g. first quarter 2024, second quarter 2024, first semester 2024) to improve retrieval.
4. A very succint high level overview (i.e. not a summary) of the chunk's content in no more than 100 characters with a focus on keywords for better similarity search retrieval

Answer only with the succinct context, and nothing else (no introduction, no conclusion, no headings).

List of company names (use exact spelling) : 
3i Group Plc
3M Company
A. O. Smith Corporation
A.P. Møller - Mærsk A/S
A2A S.p.A.
AAK
Aalberts N.V.
ABB
Abbott Laboratories
AbbVie Inc
ABN AMRO Bank N.V.
abrdn plc
Accelleron Industries AG
Accenture plc
Acciona S.A.
Accor SA
Ackermans & Van Haaren NV
ACS Actividades de Construcción y Servicios S.A.
Addtech
Adecco Group AG
adidas AG
Admiral Group plc
Adobe Inc
Advanced Micro Devices Inc
Adyen N.V.
Aedifica NV/SA
Aegon N.V.
Aena SME S.A.
Aéroports de Paris SA
Aflac Inc
ageas SA/NV
Agilent Technologies Inc
Agnico Eagle Mines Limited
AIB Group plc
Air Products and Chemicals Inc
Airbnb Inc
Airbus SE
Airtel Africa Plc
AIXTRON SE
Akamai Technologies Inc
Aker
Aker BP
Akzo Nobel N.V.
Albemarle Corp
Alcon Inc
ALD S.A.
Alexandria Real Estate Equities Inc
Alfa Laval
Align Technology Inc
Alimentation Couche-Tard Inc
Allegion plc
Allegro.eu SA
Allfunds Group PLC
Alliant Energy Corporation
Allianz SE
Ally Financial Inc
Alnylam Pharmaceuticals Inc
Alpha Services and Holdings S.A.
Alphabet Inc
Alstom SA
Alten S.A.
Altria Group Inc
Alvotech
Amadeus IT Group S.A.
Amazon.com Inc
Ambu
Amcor Plc
Amdocs Limited
Ameren Corporation
American Airlines Group Inc
American Electric Power Company Inc
American Express Company
American International Group Inc
American Tower Corporation
American Water Works Company Inc
Ameriprise Financial Inc
AMETEK Inc
Amgen Inc
Amphenol Corporation
Amplifon S.p.A.
Amundi
Analog Devices Inc
Andritz AG
Anglo American plc
AngloGold Ashanti plc
Anheuser-Busch InBev SA/NV
Annaly Capital Management Inc
ANSYS Inc
Antofagasta plc
Aon plc
APA Corporation
Apollo Global Management Inc
Apple Inc
Applied Materials Inc
Aptiv Plc
Arcadis NV
ArcelorMittal S.A.
Arch Capital Group Ltd
Archer Daniels Midland Company
Ares Management Corporation
argenx SE
Arion Banki SDB
Arista Networks Inc
Arkema S.A.
Arthur J. Gallagher & Co.
Ashtead Group PLC
ASM International NV
ASML Holding N.V.
ASR Nederland N.V.
Assa Abloy AB
Assicurazioni Generali S.p.A.
Associated British Foods plc
Assurant Inc
AstraZeneca
AT&T Inc
Atlas Copco Group
Atlassian Corporation Plc
Atmos Energy Corporation
Aurubis AG
Auto Trader Group plc
Autodesk Inc
Autohome Inc
Autoliv
Automatic Data Processing Inc
AutoStore Holdings Ltd
AutoZone Inc
AvalonBay Communities Inc
Avangrid Inc
Avantor Inc
Avanza Bank
Avery Dennison Corporation
Aviva plc
Avolta AG
AXA SA
Axfood
Axon Enterprise Inc
Azelis Group NV
Azimut Holding S.p.A.
B&M European Value Retail S.A.
Bachem Holding AG
BAE Systems plc
Baker Hughes Company
Bakkafrost
Balfour Beatty plc
Ball Corporation
Bâloise Holding AG
Banca Mediolanum S.p.A.
Banca Monte dei Paschi di Siena S.p.A.
Banca Transilvania S.A.
Banco Bilbao Vizcaya Argentaria S.A.
Banco BPM S.p.A.
Banco Comercial Português S.A.
Banco de Sabadell S.A.
Banco Santander S.A.
Bank Leumi Le-Israel B.M.
Bank of America Corporation
Bank of Ireland Group plc
Bank of Montreal
Bank Polska Kasa Opieki S.A.
Bankinter S.A.
Banque Cantonale Vaudoise
Barclays PLC
Barratt Developments plc
Barrick Gold Corporation
Barry Callebaut AG
Basf SE
Bath & Body Works Inc
BAWAG Group AG
Baxter International Inc
Bayer AG
Bayerische Motoren Werke AG
BCE Inc
BE Semiconductor Industries N.V.
Beazley plc
Bechtle AG
Becton, Dickinson and Company
Beiersdorf Aktiengesellschaft
Beijer Ref
BELIMO Holding AG
Bellway p.l.c
Berkshire Hathaway Inc
Best Buy Co Inc
Big Yellow Group Plc
BILL.com Holdings
Bio-Rad Laboratories Inc
Bio-Techne Corporation
Biogen Inc
BioMarin Pharmaceutical Inc
bioMérieux S.A
BKW AG
BlackRock Inc
Blackstone Inc
Block Inc
BNP Paribas SA
Boliden
Bolloré SE
Booking Holdings Inc
BorgWarner Inc
Boston Properties Inc
Boston Scientific Corporation
Bouygues SA
BP p.l.c.
BPER Banca SpA
Brenntag SE
Bristol-Myers Squibb Company
British American Tobacco p.l.c.
British Land Company Plc
Britvic PLC
Broadcom Inc
Broadridge Financial Solutions Inc
Brookfield Asset Management Ltd
Brookfield Corporation
Brown & Brown Inc
Brown-Forman Corporation
Brunello Cucinelli S.p.A.
BT Group plc
Bucher Industries AG
Builders FirstSource Inc
Bunge Limited
Bunzl plc
Burberry Group plc
Bureau Veritas SA
Burlington Stores Inc
Buzzi S.p.A.
C. H. Robinson Worldwide Inc
Cadence Design Systems Inc
Caesars Entertainment Inc
CaixaBank S.A.
Camden Property Trust
Cameco Corporation
Campbell Soup Company
Camtek Ltd
Canadian Imperial Bank of Commerce
Canadian National Railway Company
Canadian Natural Resources Limited
Canadian Pacific Kansas City Limited
Capgemini SE
Capital One Financial Corporation
Cardinal Health Inc
Cargotec
Carl Zeiss Meditec AG
Carlsberg Group A/S
CarMax Inc
Carnival Corporation & Plc
Carrefour SA
Carrier Global Corporation
Castellum
Catalent Inc
Caterpillar Inc
Cboe Global Markets Inc
CBRE Group Inc
CD Projekt S.A.
CDW Corporation
Celanese Corporation
Cellnex Telecom S.A.
Cencora Inc
Cenovus Energy Inc
Centene Corporation
CenterPoint Energy Inc
Centrica plc
CF Industries Holdings Inc
CGI Inc
Charles River Laboratories International Inc
Charter Communications Inc
Check Point Software Technologies Ltd
Cheniere Energy Inc
Chevron Corporation
Chipotle Mexican Grill Inc
Chocoladefabriken Lindt & Sprüngli AG
Christian Dior SE
Chubb Limited
Church & Dwight Co Inc
Cincinnati Financial Corporation
Cintas Corporation
Cisco Systems Inc
Citigroup Inc
Citizens Financial Group Inc
Clariant AG
Cloudflare Inc
CME Group Inc
CMS Energy Corporation
CNA Financial Corporation
CNH Industrial N.V.
Coca-Cola Europacific Partners PLC
Coca-Cola HBC AG
Cognizant Technology Solutions Corporation
Coinbase Global Inc
Colgate-Palmolive Company
Coloplast
Comcast Corporation
Comerica Incorporated
Comet Holding AG
Commerzbank AG
Compagnie de Saint-Gobain S.A.
Compagnie Financière Richemont SA
Compagnie Générale des Établissements Michelin Société en commandite par actions
Compass Group PLC
Computacenter PLC
Conagra Brands Inc
ConocoPhillips
Consolidated Edison Inc
Constellation Brands Inc
Constellation Energy Corporation
Constellation Software Inc
Continental AG
ConvaTec Group Plc
Copart Inc
Corning Incorporated
Corpay Inc
Corporación Acciona Energías Renovables S.A.
Corteva Inc
CoStar Group Inc
Costco Wholesale Corporation
Coterra Energy Inc
Coupang Inc
Covestro AG
Covivio S.A
Cranswick plc
Credicorp Ltd
Crédit Agricole S.A.
CRH plc
Croda International Plc
CrowdStrike Holdings Inc
Crown Castle Inc
CSX Corporation
CTP N.V.
CTS Eventim AG & Co. KGaA
Cummins Inc
CVS Health Corporation
D.R. Horton Inc
D'Ieteren Group SA
Dada Nexus Limited
Daimler Truck Holding AG
Danaher Corporation
Danone S.A.
Danske Bank
Daqo New Energy Corp
Darden Restaurants Inc
Darktrace plc
Dassault Aviation Société anonyme
Dassault Systèmes SE
Datadog Inc
Davide Campari-Milano N.V.
DaVita Inc
Dayforce Inc
DCC plc
Deckers Outdoor Corporation
Deere & Company
Delivery Hero SE
Delta Air Lines Inc
Demant
Derwent London Plc
Deutsche Bank AG
Deutsche Börse AG
Deutsche Lufthansa AG
Deutsche Post AG
Deutsche Telekom AG
Deutsche Wohnen SE
Devon Energy Corporation
Dexcom Inc
Diageo plc
Diamondback Energy Inc
DiaSorin S.p.A.
Digital Realty Trust Inc
Dino Polska S.A.
Diploma PLC
Direct Line Insurance Group plc
Discover Financial Services
DKSH Holding AG
DNB Bank
DocuSign Inc
Dollar General Corporation
Dollar Tree Inc
Dollarama Inc
Dominion Energy Inc
Domino’s Pizza Inc
DoorDash Inc
Dover Corporation
Dow Inc
Drax Group plc
DS Smith Plc
DSM Firmenich AG
DSV
DTE Energy Company
Duke Energy Corporation
DuPont De Nemours Inc
DWS Group GmbH & Co. KGaA
E.ON SE
Eastman Chemical Company
easyjet plc
Eaton Corporation plc
eBay Inc
Ecolab Inc
Edenred SA
Edison International
EDP - Energias de Portugal S.A.
EDP Renováveis S.A.
Edwards Lifesciences Corporation
Eiffage SA
Elbit Systems Ltd
Electrolux
Electronic Arts Inc
Elekta
Elevance Health Inc
Eli Lilly and Company
Elia Group SA/NV
Elis SA
Elisa
Emerson Electric Co
EMS-CHEMIE HOLDING AG
Enagás S.A.
Enbridge Inc
Endeavour Mining plc
Endesa S.A.
Enel SpA
ENGIE SA
Eni S.p.A.
Enlight Renewable Energy Ltd
Enphase Energy Inc
Entain plc
Entergy Corporation
EOG Resources Inc
EPAM Systems Inc
Epiroc
EQT
EQT Corporation
Equifax Inc
Equinix Inc
Equinor
Equitable Holdings Inc
Equity LifeStyle Properties Inc
Equity Residential
Ericsson
Erste Group Bank AG
Essex Property Trust Inc
EssilorLuxottica Société anonyme
Essity
Etsy Inc
Eurazeo SE
Eurobank Ergasias Services and Holdings S.A.
Eurofins Scientific SE
Euronext N.V.
Everest Group Ltd
Evergy Inc
Eversource Energy
Evolution
Evonik Industries AG
Evotec SE
Exact Sciences Corporation
Exelon Corporation
Exor N.V.
Expedia Group Inc
Expeditors International of Washington Inc
Experian plc
Extra Space Storage Inc
Exxon Mobil Corporation
F5 Inc
Fabege
FactSet Research Systems Inc
Fair Isaac Corporation
Fairfax Financial Holdings Limited
Fastenal Company
Fastighets AB Balder
Federal Realty Investment Trust
FedEx Corporation
Ferguson plc
Ferrari N.V
Ferrovial S.A
Fidelity National Financial Inc
Fidelity National Information Services Inc
Fielmann Aktiengesellschaft
Fifth Third Bancorp
FinecoBank Banca Fineco S.p.A.
First Citizens BancShares Inc
First Solar Inc
FirstEnergy Corp
Fiserv Inc
Flughafen Zürich AG
Flutter Entertainment plc
FMC Corporation
Ford Motor Company
Formula One Group
Fortinet Inc
Fortis Inc
Fortive Corporation
Fortnox
Fortum Corporation
Fortune Brands Innovations Inc
Forvia SE
Fox Corporation
Franco-Nevada Corporation
Franklin Resources Inc
Fraport AG
freenet AG
Freeport-McMoRan Inc
Fresenius Medical Care AG
Fresenius SE & Co. KGaA
Fresnillo plc
Frontline
Fuchs SE
Full Truck Alliance Co Ltd
Galenica AG
Galp Energia SGPS S.A.
Games Workshop Group PLC
Garmin Ltd
Gartner Inc
Gaztransport & Technigaz SA
GE HealthCare Technologies Inc
GE Vernova Inc
GEA Group Aktiengesellschaft
Geberit AG
Gecina SA
Gen Digital Inc
Generac Holdings Inc
General Dynamics Corporation
General Electric Company
General Mills Inc
General Motors Company
Genmab
Genuine Parts Company
Georg Fischer AG
George Weston Limited
Gerresheimer AG
Getinge
Getlink SE
Gilead Sciences Inc
Givaudan SA
Gjensidige Forsikring
Glanbia plc
Glencore PLC
Global Payments Inc
Globe Life Inc
GN Store Nord
GoDaddy Inc
Grafton Group plc
Great-West Lifeco Inc
Greggs plc
Grifols SA
Groupe Bruxelles Lambert SA
GSK plc
H. Lundbeck
Haleon plc
Halliburton Company
Halma plc
Handelsbanken
Hannover Rück SE
Hargreaves Lansdown plc
Hasbro Inc
Hays plc
HCA Healthcare Inc
Healthpeak Properties Inc
HEICO Corporation
Heidelberg Materials AG
Heineken Holding N.V.
Heineken N.V.
Hellenic Telecommunications Organization S.A.
HelloFresh SE
Helvetia Holding AG
Hemnet Group
Henkel AG & Co. KGaA
Hennes & Mauritz
Henry Schein Inc
Hera S.p.A.
Hermès International Société en commandite par actions
Hess Corporation
Hewlett Packard Enterprise Company
Hexagon
HEXPOL
Hikma Pharmaceuticals PLC
Hilton Worldwide Holdings Inc
Hiscox Ltd
HOCHTIEF Aktiengesellschaft
Holcim Ltd
Holmen
Hologic Inc
Honeywell International Inc
Hormel Foods Corporation
Host Hotels & Resorts Inc
Howden Joinery Group Plc
Howmet Aerospace Inc
HP Inc
HSBC Holdings plc
Hubbell Inc
HubSpot Inc
Hugo Boss AG
Huhtamäki
Humana Inc
Huntington Bancshares Incorporated
Huntington Ingalls Industries Inc
Husqvarna
Hydro One Limited
Iberdrola S.A.
ICL Group Ltd
ICON Public Limited Company
IDEX Corporation
IDEXX Laboratories Inc
IG Group Holdings plc
Illinois Tool Works Inc
Illumina Inc
IMCD N.V.
IMI plc
Imperial Brands PLC
Imperial Oil Limited
Inchcape plc
Incyte Corporation
Indivior PLC
Industria de Diseño Textil S.A.
Industrivärden
Indutrade
INFICON Holding AG
Infineon Technologies AG
Informa plc
Infrastrutture Wireless Italiane S.p.A.
ING Groep N.V.
Ingersoll Rand Inc
Insulet Corporation
Intact Financial Corporation
Intel Corporation
Intercontinental Exchange Inc
InterContinental Hotels Group PLC
Intermediate Capital Group plc
International Business Machines Corp
International Consolidated Airlines Group S.A.
International Distributions Services plc
International Flavors & Fragrances Inc
International Paper Company
Interpump Group S.p.A.
Intertek Group plc
Intesa Sanpaolo S.p.A.
Intuit Inc
Intuitive Surgical Inc
Invesco Ltd
Investor
Invitation Homes Inc
Ipsen S.A.
Ipsos SA
iQIYI Inc
IQVIA Holdings Inc
Iron Mountain Incorporated
Israel Discount Bank Limited
ISS
Italgas S.p.A.
ITV plc
Iveco Group N.V.
J Sainsbury plc
J.B. Hunt Transport Services Inc
Jabil Inc
Jack Henry & Associates Inc
Jacobs Solutions Inc
Jazz Pharmaceuticals plc
JCDecaux SE
JD Sports Fashion plc
JDE Peet's N.V.
Jerónimo Martins SGPS S.A.
Johnson & Johnson
Johnson Controls International plc
Johnson Matthey Plc
JOYY Inc
JPMorgan Chase & Co
Julius Bär Gruppe AG
Jumbo S.A.
Juniper Networks Inc
Just Eat Takeaway.com N.V.
Jyske Bank
K+S Aktiengesellschaft
Kanzhun Limited
KBC Group NV
KE Holdings Inc
Kellanova
Kenvue Inc
Kering SA
Kerry Group PLC
Kesko
Keurig Dr Pepper Inc
KeyCorp
Keysight Technologies Inc
KGHM Polska Miedz S.A.
Kimberly-Clark Corporation
Kimco Realty Corporation
Kinder Morgan Inc
Kingfisher plc
Kingspan Group Plc
Kion Group AG
KKR & Co Inc
KLA Corporation
Klépierre
Knorr-Bremse Aktiengesellschaft
Kojamo
KONE
Konecranes
Kongsberg Gruppen
Koninklijke Ahold Delhaize N.V.
Koninklijke KPN N.V.
Koninklijke Philips N.V.
Koninklijke Vopak N.V.
Kuehne + Nagel International AG
L'Air Liquide S.A.
L'Occitane International S.A.
L'Oréal SA
L3harris Technologies Inc
La Française des Jeux Société anonyme
Laboratory Corporation of America Holdings
Lagercrantz Group
Lam Research Corporation
Lamb Weston Holdings Inc
Land Securities Group plc
Lanxess AG
Las Vegas Sands Corp
Latour
Lear Corporation
LEG Immobilien AG
Legal & General Group Plc
Legrand SA
Leidos Holdings Inc
Lennar Corporation
Leonardo S.p.a.
Liberty Broadband Corporation
Liberty Global plc
Lifco
Linde plc
Live Nation Entertainment Inc
LKQ Corporation
Lloyds Banking Group plc
Loblaw Companies Limited
Lockheed Martin Corporation
Loews Corporation
Logitech International S.A.
London Stock Exchange Group Plc
Londonmetric Property PLC
Lonza Group AG
Lotus Bakeries NV
Lowe’s Companies Inc
LPL Financial Holdings Inc
LPP SA
Lucid Group Inc
Lufax Holding Ltd
Lululemon Athletica Inc
Lundbergföretagen
LVMH Moët Hennessy - Louis Vuitton, Société Européenne
LyondellBasell Industries N.V.
M&G plc
M&T Bank Corporation
Magna International Inc
Man Group Plc
Manulife Financial Corporation
Mapfre S.A.
Marathon Oil Corporation
Marathon Petroleum Corporation
Marel hf.
Markel Group Inc
MarketAxess Holdings Inc
Marks & Spencer Group Plc
Marriott International Inc
Marsh & McLennan Companies Inc
Martin Marietta Materials Inc
Marvell Technology Inc
Masco Corporation
Mastercard Inc
Match Group Inc
McCormick & Company Incorporated
McDonald’s Corporation
McKesson Corporation
Mediobanca Banca di Credito Finanziario S.p.A.
Medtronic plc
Melrose Industries PLC
MercadoLibre Inc
Mercedes-Benz Group AG
Merck & Co Inc
Merck KGaA
MERLIN Properties SOCIMI S.A.
Meta Platforms Inc
MetLife Inc
Metro Inc
Metso
Mettler-Toledo International Inc
MGM Resorts International
Microchip Technology Inc
Micron Technology Inc
Microsoft Corporation
Mid-America Apartment Communities Inc
Millicom International Cellular
Moderna Inc
Mohawk Industries Inc
Molina Healthcare Inc
Molson Coors Beverage Company
Moncler S.p.A.
Mondelez International Inc
Mondi plc
MongoDB Inc
Monolithic Power Systems Inc
Monster Beverage Corporation
Moody’s Corporation
Morgan Stanley
Motorola Solutions Inc
Mowi
MSCI Inc
MTU Aero Engines AG
Münchener Rückversicherungs-Gesellschaft Aktiengesellschaft in München
Nasdaq Inc
National Bank of Canada
National Bank of Greece S.A.
National Grid PLC
Naturgy Energy Group S.A.
NatWest Group plc
Nemetschek SE
Neoen S.A.
Neste Oyj
Nestlé S.A.
NetApp Inc
Netflix Inc
Newmont Corporation
News Corporation
Nexans S.A.
Nexi S.p.A.
NEXT plc
NextEra Energy Inc
Nibe Industrier
NICE Ltd
Nike Inc
NiSource Inc
NKT
NN Group N.V.
Nokia Oyj
Nordea Bank Abp
Nordnet
Nordson Corporation
Norfolk Southern Corporation
Norsk Hydro
Northern Trust Corporation
Northrop Grumman Corporation
Norwegian Cruise Line Holdings Ltd
Nova Ltd
Novartis AG
Novo Nordisk
Novonesis
NRG Energy Inc
Nucor Corporation
Nutrien Ltd
NVIDIA Corporation
NVR Inc
NXP Semiconductors N.V.
O’Reilly Automotive Inc
Ocado Group plc
Occidental Petroleum Corporation
OCI N.V.
Okta Inc
Old Dominion Freight Line Inc
Omnicom Group Inc
OMV Aktiengesellschaft
ON Semiconductor Corporation
ONEOK Inc
Oracle Corporation
Orange S.A.
Orion Oyj
Orkla
Orlen S.A.
Ørsted
Otis Worldwide Corporation
OTP Bank Nyrt
Paccar Inc
Packaging Corporation of America
Palantir Technologies Inc
Palo Alto Networks Inc
Pandora
Paramount Global
Parker-Hannifin Corporation
Partners Group Holding AG
Paychex Inc
Paycom Software Inc
PayPal Holdings Inc
PDD Holdings Inc
Pearson plc
Pembina Pipeline Corporation
Pennon Group plc
Pentair plc
Pepco Group N.V.
PepsiCo Inc
Pernod Ricard SA
Persimmon Plc
Pfizer Inc
PG&E Corporation
Philip Morris International Inc
Phillips 66
Phoenix Group Holdings plc
Pinnacle West Capital Corporation
Pinterest Inc
Piraeus Financial Holdings S.A.
Pirelli & C. S.p.A.
Pluxee N.V.
Pool Corporation
Porsche AG
Porsche Automobil Holding SE
Poste Italiane SpA
Power Corporation of Canada
Powszechna Kasa Oszczednosci Bank Polski Spólka Akcyjna
Powszechny Zakład Ubezpieczeń
PPG Industries Inc
PPL Corporation
Prada S.p.A.
Principal Financial Group Inc
Prologis Inc
Prosus N.V.
Prudential Financial Inc
Prudential plc
Prysmian S.p.A.
PSP Swiss Property AG
PTC Inc
Public Service Enterprise Group Inc
Public Storage
Publicis Groupe S.A.
PulteGroup Inc
PUMA SE
Qatar National Bank (Q.P.S.C.)
Qiagen N.V.
Qifu Technology Inc
QinetiQ Group PLC
Qorvo Inc
Qualcomm Incorporated
Quanta Services Inc
Quartr name
Quest Diagnostics Inc
Raiffeisen Bank International AG
Ralph Lauren Corporation
Randstad N.V.
RATIONAL Aktiengesellschaft
Raymond James Financial Inc
Realty Income Corporation
Reckitt Benckiser Group plc
Recordati Industria Chimica e Farmaceutica S.p.A.
Redeia Corporación S.A.
Regency Centers Corporation
Regeneron Pharmaceuticals Inc
Regions Financial Corporation
RELX PLC
Rémy Cointreau SA
RenaissanceRe Holdings Ltd
Renault SA
Rentokil Initial plc
Reply S.p.A.
Repsol S.A.
Republic Services Inc
ResMed Inc
Restaurant Brands International Inc
Revvity Inc
Rexel S.A.
Rheinmetall AG
Richter Gedeon Vegyészeti Gyár Nyilvánosan Muködo Rt
Rightmove plc
Ringkjøbing Landbobank
Rivian Automotive Inc
RLX Technology Inc
Robert Half Inc
Roblox Corporation
Roche Holding AG
Rockwell Automation Inc
Rockwool A/S
Rollins Inc
Rolls-Royce Holdings plc
Roper Technologies Inc
Ross Stores Inc
Rotork plc
Royal Bank of Canada
Royal Caribbean Cruises Ltd
Royal Unibrew
Royalty Pharma plc
RS Group plc
RTL Group S.A.
RTX Corporation
Rubis
RWE Aktiengesellschaft
Ryanair Holdings Plc
S.P.E.E.H. Hidroelectrica S.A.
S&P Global Inc
SAAB
Safestore Holdings PLC
Safran SA
Sagax
Salesforce Inc
SalMar
Sampo
Samsonite International S.A.
Sandoz Group AG
Sandvik
Sanofi
Santander Bank Polska S.A.
SAP SE
Saputo Inc
Sartorius Aktiengesellschaft
Sartorius Stedim Biotech S.A.
SBA Communications Corporation
SCA
Schibsted
Schindler Holding AG
Schlumberger Limited
Schneider Electric S.E.
Schroders plc
SCOR SE
Scout24 SE
Seagate Technology Holdings plc
SEB SA
SECTRA
Securitas
SEGRO Plc
SEI Investments Co
Sempra
Serco Group plc
ServiceNow Inc
Severn Trent Plc
SGS SA
Shell plc
Shopify Inc
Siegfried Holding AG
Siemens AG
Siemens Energy AG
Siemens Healthineers AG
SIG Group AG
Signify N.V.
Sika AG
Simon Property Group Inc
Sirius XM Holdings Inc
Sixt SE
Skandinaviska Enskilda Banken
Skanska
SKF
Skyworks Solutions Inc
Smith & Nephew plc
Smiths Group plc
Smurfit Kappa Group plc
Snam S.p.A.
Snap-on Incorporated
Snowflake Inc
Société Générale Société anonyme
Sodexo S.A.
Sofina Société Anonyme
Softcat plc
Soitec S.A.
Solvay SA
Solventum Corporation
Sonova Holding AG
Sopra Steria Group SA
Southern Copper Corporation
Southwest Airlines Co
Spectris plc
SPIE SA
Spirax-Sarco Engineering plc
Spotify Technology S.A.
SS&C Technologies Holdings Inc
SSAB
SSE plc
SSP Group plc
St. James's Place plc
Standard Chartered PLC
Stanley Black & Decker Inc
Starbucks Corporation
State Bank of India
State Street Corporation
Steel Dynamics Inc
Stellantis N.V.
STERIS plc
STMicroelectronics N.V.
Stora Enso
Storebrand
Straumann Holding AG
Stryker Corporation
Subsea 7
Sun Communities Inc
Sun Life Financial Inc
Suncor Energy Inc
Super Micro Computer Inc
Svitzer Group
Sweco
Swedbank
Swedish Orphan Biovitrum
Swiss Life Holding AG
Swiss Prime Site AG
Swiss Re AG
Swisscom AG
Swissquote Group Holding Ltd
Sydbank
Syensqo SA/NV
Symrise AG
Synchrony Financial
Synopsys Inc
Sysco Corporation
T-Mobile US Inc
T. Rowe Price Group Inc
TAG Immobilien AG
Take-Two Interactive Software Inc
Talanx AG
Tapestry Inc
Targa Resources Corp
Target Corporation
Tate & Lyle plc
Taylor Wimpey plc
TC Energy Corporation
TE Connectivity Ltd
TeamViewer AG
Tecan Group AG
Technip Energies N.V.
Teck Resources Ltd
Tele2
Telecom Italia S.p.A.
Teledyne Technologies Incorporated
Teleflex Incorporated
Telefónica S.A.
Telekom Austria AG
Telenor
Teleperformance SE
Telia Company
TELUS Corporation
Temenos AG
Tenaris S.A.
Tencent Music Entertainment Group
Teradyne Inc
TERNA ENERGY Industrial Commercial Technical Societe Anonyme
Terna S.p.A
Tesco PLC
Tesla Inc
Teva Pharmaceutical Industries Limited
Texas Instruments Incorporated
Texas Pacific Land Corporation
Textron Inc
Thales S.A.
The AES Corporation
The Allstate Corporation
The Bank of New York Mellon Corporation
The Bank of Nova Scotia
The Berkeley Group Holdings plc
The Boeing Company
The Carlyle Group Inc
The Charles Schwab Corporation
The Cigna Group
The Clorox Company
The Coca-Cola Company
The Cooper Companies Inc
The Estée Lauder Companies Inc
The Goldman Sachs Group Inc
The Hartford Financial Services Group Inc
The Hershey Company
The Home Depot Inc
The Interpublic Group of Companies Inc
The J. M. Smucker Company
The Kraft Heinz Company
The Kroger Co
The Mosaic Company
The PNC Financial Services Group Inc
The Procter & Gamble Company
The Progressive Corporation
The Sage Group plc
The Sherwin-Williams Company
The Southern Company
The Swatch Group AG
The Toronto-Dominion Bank
The Trade Desk Inc
The Travelers Companies Inc
The Walt Disney Company
The Weir Group PLC
The Williams Companies Inc
Thermo Fisher Scientific Inc
Thomson Reuters Corporation
Thule
thyssenkrupp AG
TietoEVRY
Titan Cement International S.A.
Tomra Systems
Topdanmark
TotalEnergies SE
Tourmaline Oil Corp
Tower Semiconductor Ltd
Tractor Supply Company
Trane Technologies plc
TransDigm Group Incorporated
TransUnion
TRATON
Travis Perkins plc
Trelleborg
Trimble Inc
Tritax Big Box REIT PLC
Truist Financial Corporation
Tryg
Tuya Inc
Twilio Inc
Tyler Technologies Inc
Tyson Foods Inc
U.S. Bancorp
Uber Technologies Inc
Ubiquiti Inc
Ubisoft Entertainment SA
UBS Group AG
UCB SA
UDR Inc
UiPath Inc
Ulta Beauty Inc
Umicore SA
Unibail-Rodamco-Westfield SE
UniCredit S.p.A.
Unilever PLC
Union Pacific Corporation
Unipol Gruppo S.p.A.
Unite Group Plc
United Airlines Holdings Inc
United Parcel Service Inc
United Rentals Inc
United Utilities Group PLC
UnitedHealth Group Incorporated
Unity Software Inc
Universal Health Services Inc
Universal Music Group N.V.
UPM-Kymmene Corporation
Vail Resorts Inc
Valeo SE
Valero Energy Corporation
Vallourec S.A.
Valmet
Vår Energi AS
VAT Group AG
Veeva Systems Inc
Ventas Inc
Veolia Environnement S.A.
Verallia Société Anonyme
Veralto Corporation
VERBUND AG
VeriSign Inc
Verisk Analytics Inc
Verizon Communications Inc
Vertex Pharmaceuticals Inc
Vestas Wind Systems
Viatris Inc
VICI Properties Inc
Vidrala S.A.
Vinci SA
Vipshop Holdings Limited
Virgin Money UK PLC
Visa Inc
Viscofan S.A.
Vistra Corp
Vistry Group PLC
Vivendi SE
Vodafone Group PLC
Voestalpine AG
Volkswagen AG
Volvo
Volvo Car
Vonovia SE
Voya Financial Inc
Vulcan Materials Company
W W Grainger Inc
W. P. Carey Inc
W. R. Berkley Corporation
Wacker Chemie AG
Walgreens Boots Alliance Inc
Wallenstam
Walmart Inc
Warehouses De Pauw NV/SA
Warner Bros. Discovery Inc
Wärtsilä
Waste Connections Inc
Waste Management Inc
Waters Corporation
WEC Energy Group Inc
Weibo Corporation
Wells Fargo & Company
Welltower Inc
Wendel
West Pharmaceutical Services Inc
Western Digital Corporation
Westinghouse Air Brake Technologies Corporation
Westlake Corp
WestRock Company
Weyerhaeuser Company
WH Smith plc
Wheaton Precious Metals Corp
Whitbread PLC
Wienerberger AG
Wihlborgs Fastigheter
Willis Towers Watson plc
Wise plc
Wolters Kluwer N.V.
Workday Inc
Worldline SA
WPP plc
Wynn Resorts Limited
Xcel Energy Inc
Xylem Inc
Yara International
Yum China Holdings Inc
Yum! Brands Inc
Zalando SE
Zealand Pharma
Zebra Technologies Corporation
Zillow Group Inc
Zimmer Biomet Holdings Inc
Zoetis Inc
Zoom Video Communications Inc
Zscaler Inc
Zurich Insurance Group AG"""

def load_processed_urls():
    try:
        with open(STATE_FILE) as f:
            return set(json.load(f))
    except FileNotFoundError:
        return set()

def save_processed_urls(urls):
    Path(STATE_FILE).parent.mkdir(exist_ok=True)
    with open(STATE_FILE, 'w') as f:
        json.dump(list(urls), f)

if 'processed_urls' not in st.session_state:
    st.session_state.processed_urls = load_processed_urls()

if 'processing_metrics' not in st.session_state:
    st.session_state.processing_metrics = {
        'total_documents': 0,
        'processed_documents': 0,
        'total_chunks': 0,
        'successful_chunks': 0,
        'failed_chunks': 0,
        'cache_hits': 0,
        'start_time': None,
        'errors': []
    }

# Create a persistent temp directory
TEMP_DIR = Path("./.temp")
TEMP_DIR.mkdir(exist_ok=True)

# Client initialization
with st.expander("Client Initialization", expanded=True):
    try:
        client = anthropic.Client(
            api_key=st.secrets['ANTHROPIC_API_KEY'],
            default_headers={"anthropic-beta": "prompt-caching-2024-07-31"}
        )
        # Initialize LlamaParse with only supported parameters
        llama_parser = LlamaParse(
            api_key=st.secrets['LLAMA_PARSE_API_KEY'],
            result_type="text"
        )
        embed_model = VoyageEmbedding(
            model_name="voyage-finance-2",
            voyage_api_key=st.secrets['VOYAGE_API_KEY']
        )
        st.success("✅ All clients initialized successfully")
    except Exception as e:
        st.error(f"❌ Error initializing clients: {str(e)}")
        st.stop()

# Configuration section
with st.expander("Processing Configuration", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.number_input("Chunk Size", value=1000, min_value=100, max_value=4000)
        chunk_overlap = st.number_input("Chunk Overlap", value=200, min_value=0, max_value=1000)
        model = st.selectbox(
            "Claude Model",
            options=[
                "claude-3-haiku-20240307",
                "claude-3-sonnet-20240229",
                "claude-3-opus-20240229"
            ],
            index=0,
            help="Select the Claude model to use for processing"
        )
    with col2:
        context_prompt = st.text_area(
            "Context Prompt",
            value=DEFAULT_PROMPT,
            height=200,
            help="Customize the prompt for context generation"
        )
        force_reprocess = st.checkbox("Force Reprocess All")
        if st.button("Reset Processing State"):
            st.session_state.processed_urls = set()
            save_processed_urls(st.session_state.processed_urls)
            st.success("Processing state reset")
            st.rerun()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=5, min=5, max=60),
    retry=retry_if_exception(lambda e: "overloaded_error" in str(e))
)
def get_chunk_context(client, chunk: str, full_doc: str, system_prompt: str, model: str):
    """Get context with retry logic for overload errors"""
    try:
        context = client.messages.create(
            model=model,
            max_tokens=200,
            system=system_prompt,
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": "<document>\n" + full_doc + "\n</document>",
                            "cache_control": {"type": "ephemeral"}
                        },
                        {
                            "type": "text",
                            "text": "\nHere is the chunk we want to situate within the whole document:\n<chunk>\n" + chunk + "\n</chunk>"
                        }
                    ]
                }
            ]
        )
        return context
    except Exception as e:
        if "overloaded_error" in str(e):
            st.warning(f"Claude is overloaded, retrying in a few seconds...")
            raise e
        raise e

def process_document(url: str, metrics: dict, model: str, context_prompt: str) -> bool:
    try:
        filename = unquote(url.split('/')[-1])
        st.write(f"Downloading {filename}...")
        
        pdf_response = requests.get(url, timeout=30)
        pdf_response.raise_for_status()
        
        # Save to persistent temp directory with unique name
        temp_path = TEMP_DIR / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
        
        with open(temp_path, 'wb') as f:
            f.write(pdf_response.content)
        
        try:
            st.write("Parsing document...")
            st.write(f"PDF file size: {os.path.getsize(temp_path)} bytes")
            
            # Try parsing with simplified call
            parsed_docs = llama_parser.load_data(str(temp_path))
            
            if not parsed_docs:
                st.warning(f"No sections found in document: {filename}")
                return False
                
            st.write(f"Found {len(parsed_docs)} sections")
            
            for doc in parsed_docs:
                # Get full document text for context
                full_doc_text = doc.text
                st.write(f"Full document length: {len(full_doc_text)} characters")
                
                chunks = []
                current_chunk = []
                current_length = 0
                
                for line in doc.text.split('\n'):
                    line_length = len(line)
                    if current_length + line_length > chunk_size and current_chunk:
                        chunks.append('\n'.join(current_chunk))
                        overlap_text = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk
                        current_chunk = overlap_text
                        current_length = sum(len(line) for line in current_chunk)
                    
                    current_chunk.append(line)
                    current_length += line_length
                
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                
                metrics['total_chunks'] += len(chunks)
                st.write(f"Created {len(chunks)} chunks")
                
                chunk_progress = st.progress(0)
                for i, chunk in enumerate(chunks):
                    try:
                        st.write(f"Processing chunk {i+1}/{len(chunks)}...")
                        context = get_chunk_context(
                            client=client,
                            chunk=chunk,
                            full_doc=full_doc_text,
                            system_prompt=context_prompt,
                            model=model
                        )
                        
                        embedding = embed_model.get_text_embedding(chunk)
                        
                        metrics['successful_chunks'] += 1
                        chunk_progress.progress((i + 1) / len(chunks))
                        
                        with st.expander(f"Chunk {i+1} Results", expanded=False):
                            st.write("Context:", context.content[0].text)
                            st.write("Embedding size:", len(embedding))
                            
                        if hasattr(context, 'usage') and hasattr(context.usage, 'cache_read_input_tokens'):
                            if context.usage.cache_read_input_tokens > 0:
                                metrics['cache_hits'] += 1
                        
                    except Exception as e:
                        metrics['failed_chunks'] += 1
                        metrics['errors'].append(f"Chunk processing error in {url}: {str(e)}")
                        st.error(f"Error processing chunk: {str(e)}")
                        continue
            
            return True
            
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
    except Exception as e:
        metrics['errors'].append(f"Document processing error for {url}: {str(e)}")
        st.error(f"Error processing document: {str(e)}")
        return False

# Main UI section
st.title("PDF Processing Pipeline")
st.subheader("Process PDFs from Sitemap")

sitemap_url = st.text_input(
    "Enter Sitemap URL",
    value="https://alpinedatalake7.s3.eu-west-3.amazonaws.com/sitemap.xml"
)

if st.button("Start Processing"):
    try:
        st.session_state.processing_metrics = {
            'total_documents': 0,
            'processed_documents': 0,
            'total_chunks': 0,
            'successful_chunks': 0,
            'failed_chunks': 0,
            'cache_hits': 0,
            'start_time': datetime.now(),
            'errors': []
        }
        
        st.write("Fetching sitemap...")
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        
        namespaces = {
            None: "",
            "ns": "http://www.sitemaps.org/schemas/sitemap/0.9"
        }
        
        pdf_urls = []
        for ns in namespaces.values():
            if ns:
                urls = root.findall(f".//{{{ns}}}loc")
            else:
                urls = root.findall(".//loc")
            
            pdf_urls.extend([url.text for url in urls if url.text.lower().endswith('.pdf')])
            if pdf_urls:
                break
        
        if not pdf_urls:
            st.error("No PDF URLs found in sitemap")
            st.code(response.text, language="xml")
            st.stop()
            
        st.write("Found PDFs:", pdf_urls)
        
        if not force_reprocess:
            new_urls = [url for url in pdf_urls if url not in st.session_state.processed_urls]
            skipped = len(pdf_urls) - len(new_urls)
            if skipped > 0:
                st.info(f"Skipping {skipped} previously processed documents")
            pdf_urls = new_urls
        
        if not pdf_urls:
            st.success("No new documents to process!")
            st.stop()
        
        st.session_state.processing_metrics['total_documents'] = len(pdf_urls)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        metrics_cols = st.columns(4)
        
        for i, url in enumerate(pdf_urls):
            status_text.text(f"Processing document {i+1}/{len(pdf_urls)}: {unquote(url.split('/')[-1])}")
            
            success = process_document(
                url=url,
                metrics=st.session_state.processing_metrics,
                model=model,
                context_prompt=context_prompt
            )
            if success:
                st.session_state.processing_metrics['processed_documents'] += 1
                st.session_state.processed_urls.add(url)
            
            progress_bar.progress((i + 1) / len(pdf_urls))
            
            with metrics_cols[0]:
                st.metric("Documents Processed", f"{st.session_state.processing_metrics['processed_documents']}/{st.session_state.processing_metrics['total_documents']}")
            with metrics_cols[1]:
                st.metric("Chunks Processed", st.session_state.processing_metrics['successful_chunks'])
            with metrics_cols[2]:
                st.metric("Cache Hits", st.session_state.processing_metrics['cache_hits'])
            with metrics_cols[3]:
                elapsed = datetime.now() - st.session_state.processing_metrics['start_time']
                st.metric("Processing Time", f"{elapsed.total_seconds():.1f}s")
        
        save_processed_urls(st.session_state.processed_urls)
        
        st.success(f"""
            Processing complete!
            - Documents processed: {st.session_state.processing_metrics['processed_documents']}/{st.session_state.processing_metrics['total_documents']}
            - Successful chunks: {st.session_state.processing_metrics['successful_chunks']}
            - Failed chunks: {st.session_state.processing_metrics['failed_chunks']}
            - Cache hits: {st.session_state.processing_metrics['cache_hits']}
            - Total time: {(datetime.now() - st.session_state.processing_metrics['start_time']).total_seconds():.1f}s
        """)
        
        if st.session_state.processing_metrics['errors']:
            with st.expander("Show Errors", expanded=False):
                for error in st.session_state.processing_metrics['errors']:
                    st.error(error)
                    
    except Exception as e:
        st.error(f"Error processing sitemap: {str(e)}")

with st.expander("Current Processing State", expanded=False):
    st.write(f"Previously processed URLs: {len(st.session_state.processed_urls)}")
    if st.session_state.processed_urls:
        for url in sorted(st.session_state.processed_urls):
            st.write(f"- {unquote(url.split('/')[-1])}")

# Cleanup temp directory on exit
for file in TEMP_DIR.glob("*"):
    try:
        os.remove(file)
    except:
        pass







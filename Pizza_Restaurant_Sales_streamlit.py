import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import ast
import json
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import matplotlib.patches as mpatches

df = pd.read_excel(r'C:\Users\ARIEH\Downloads\Data Model - Pizza Sales.xlsx')

# Afficher les informations de base sur le DataFrame
st.title("Ventes de pizza du restaurant:")

# Ajoutez une option de sélection dans la barre latérale pour chaque section
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Aller à", ["Chapitre 1 : Exploration", "Chapitre 2 : Analyse", "Conclusion"])


if selection == "Chapitre 1 : Exploration":
    st.header("Chapitre 1 : Exploration")

    st.write("""Rappel du contexte Pour le Maven Pizza Challenge,je joue le rôle d'un consultant BI embauché par Plato's Pizza, une pizzeria d'inspiration grecque du New Jersey. j'ai été embauché pour aider le restaurant à utiliser les données pour améliorer ses opérations, et vous venez de recevoir la note suivante :

Bienvenue à bord, nous sommes ravis que vous soyez là pour vous aider !

Les choses vont bien ici chez Platon, mais il y a place à l'amélioration. Nous avons collecté des données transactionnelles au cours de la dernière année, mais nous n'avons vraiment pas été en mesure de les utiliser à bon escient. En espérant que vous pourrez analyser les données et créer un rapport pour nous aider à trouver des opportunités de générer plus de ventes et de travailler plus efficacement.

Voici quelques questions auxquelles nous aimerions pouvoir répondre :

Quels jours et heures avons-nous tendance à être les plus occupés ?
Combien de pizzas faisons-nous pendant les périodes de pointe ?
Quelles sont nos pizzas les plus et les moins vendues ?
Quelle est notre valeur moyenne de commande ?
Dans quelle mesure utilisons-nous notre capacité en sièges ? (nous avons 15 tables et 60 places)
Quelle est la relation entre le nombre de pizzas vendues et le chiffre d’affaires ? (Cela pourrait aider à comprendre si la vente de certaines pizzas contribue plus au chiffre d’affaires.)
Quels sont les types de pizzas les plus populaires pendant les différentes périodes de la journée ?( Cela pourrait aider à optimiser le menu pour différentes périodes de la journée.)
C'est tout ce à quoi je peux penser pour l'instant, mais si vous avez d'autres idées, j'aimerais les entendre - vous êtes l'expert !

Merci d'avance,

Mario Maven (gérant, Plato's Pizza)""")
    
    st.write("""
    - **order_id**: Identifiant unique pour chaque commande passée par une table
    - **order_details_id**: Identifiant unique pour chaque pizza commandée dans chaque commande (les pizzas du même type et de la même taille sont conservées dans la même ligne, et la quantité augmente)
    - **pizza_id**: Identifiant de clé unique qui relie la pizza commandée à ses détails, comme la taille et le prix
    - **quantity**: Quantité commandée pour chaque pizza du même type et de la même taille
    - **order_date**: Date à laquelle la commande a été passée (entrée dans le système avant la cuisson et le service)
    - **order_time**: Heure à laquelle la commande a été passée (entrée dans le système avant la cuisson et le service)
    - **unit_price**: Prix de la pizza en USD
    - **total_price**: unit_price * quantity
    - **pizza_size**: Taille de la pizza (Small, Medium, Large, X Large, ou XX Large)
    - **pizza_type**: Identifiant de clé unique qui relie la pizza commandée à ses détails, comme la taille et le prix
    - **pizza_ingredients**: ingrédients utilisés dans la pizza tels qu'ils apparaissent dans le menu (ils contiennent tous du fromage mozzarella, même si ce n'est pas spécifié ; et ils contiennent tous de la sauce tomate, sauf si une autre sauce est spécifiée)
    - **pizza_name**: Nom de la pizza tel qu'il apparaît dans le menu
    """)


    # Calculer le nombre unique de choix de pizzas
    num_pizza_choices = df['pizza_name'].nunique()

    # Calculer le nombre unique de tailles de pizzas
    num_pizza_sizes = df['pizza_size'].nunique()

    # Afficher les résultats
    st.write(f"Il y a {num_pizza_choices} choix de pizzas et {num_pizza_sizes} tailles de pizzas différentes dans ce restaurant.")

    def calculer_prix_min_max(df):
        """Calcule le prix minimum et maximum des pizzas."""
        min_price = df['unit_price'].min()
        max_price = df['unit_price'].max()
        return min_price, max_price

    def afficher_prix(min_price, max_price):
        """Affiche le prix minimum et maximum des pizzas."""
        st.write(f"Le prix des pizzas varie de : ${min_price:.2f} - ${max_price:.2f}")

    # Utiliser les fonctions définies
    min_price, max_price = calculer_prix_min_max(df)
    afficher_prix(min_price, max_price)


    # Sélectionner les colonnes pertinentes
    menu_data = df[['pizza_name', 'pizza_size', 'unit_price']]

    # Regrouper les données par nom de pizza et taille
    menu = menu_data.groupby(['pizza_name', 'pizza_size'])['unit_price'].mean().reset_index()

    
    # Créer la carte du restaurant
    st.title("Carte du restaurant")
    for pizza_name, group in menu.groupby('pizza_name'):
        st.header(pizza_name)
        st.table(group.set_index('pizza_size')['unit_price'].rename('Prix (USD)'))

    def calculer_prix_min_max(df):
        """Calcule le prix minimum et maximum des pizzas."""
        min_price = df['total_price'].min()
        max_price = df['total_price'].max()
        return min_price, max_price

    def afficher_prix(min_price, max_price):
        """Affiche le prix minimum et maximum des pizzas."""
        st.write(f"Le prix des pizzas varie de : ${min_price:.2f} - ${max_price:.2f}")

    # Utiliser les fonctions définies
    min_price, max_price = calculer_prix_min_max(df)
    afficher_prix(min_price, max_price)


    

elif selection == "Chapitre 2 : Analyse":
    st.header("Chapitre 2 : Analyse")

 
    df['Day_of_week']= df['order_date'].dt.day_name()
    df['Day_of_week_no']= df['order_date'].dt.day_of_week
    df['Month']= df['order_date'].dt.month_name()
    df['Month_no']= df['order_date'].dt.month
    df['order_time']=df['order_time'].astype('string')
    df[['Hour','Minute', 'Second']]=df['order_time'].str.split(":",expand=True)

    # Sous-partie 1
    st.subheader("Quels mois, jours et heures avons-nous tendance à être les plus occupés ?")

    # Regrouper les données par mois et calculer la somme des quantités
    month = df.groupby(['Month', 'Month_no'])['quantity'].sum().reset_index().sort_values(['Month_no'])

    # Créer une figure
    fig = go.Figure()

    # Ajouter une trace de ligne
    fig.add_trace(go.Scatter(x=month["Month"], y=month["quantity"], mode='lines'))

    # Mettre à jour le layout pour avoir un fond blanc
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # Ajouter des titres
    fig.update_layout(title='Quantité de pizzas vendues par mois',
                    xaxis_title='Mois',
                    yaxis_title='Quantité de pizzas vendues')

    # Afficher le graphique
    st.plotly_chart(fig)

    week= df.groupby(['Day_of_week', 'Day_of_week_no'])['quantity'].sum().reset_index().sort_values(['Day_of_week_no'])
    fig = px.line(week, x="Day_of_week", y="quantity")

    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # Ajouter des titres
    fig.update_layout(title='Quantité de pizzas vendues par jour de la semaine',
                    xaxis_title='Jour de la semaine',
                    yaxis_title='Quantité de pizzas vendues')

    st.plotly_chart(fig)

    # Grouper par heure et calculer la somme de la quantité de pizzas vendues
    hr = df.groupby('Hour')['quantity'].sum().reset_index()

    # Créer une figure
    fig = go.Figure()

    # Ajouter une trace de ligne
    fig.add_trace(go.Scatter(x=hr["Hour"], y=hr["quantity"], mode='lines'))

    # Mettre à jour le layout pour avoir un fond blanc
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # Ajouter des titres
    fig.update_layout(title='Quantité de pizzas vendues par heure',
                    xaxis_title='Heure',
                    yaxis_title='Quantité de pizzas vendues')

    # Afficher le graphique
    st.plotly_chart(fig)



    # Sous-partie 2
    st.subheader("Combien de pizzas faisons-nous pendant les périodes de pointe ?")

    # Regrouper les données par heure et calculer la somme des quantités
    hourly_sales = df.groupby('Hour')['quantity'].sum().reset_index()

    # Trouver l'heure avec la quantité maximale de pizzas vendues
    peak_hour = hourly_sales[hourly_sales['quantity'] == hourly_sales['quantity'].max()]

    # Afficher les résultats
    st.write(f"La période de pointe pour la vente de pizzas est à {peak_hour['Hour'].values[0]} heure, avec {peak_hour['quantity'].values[0]} pizzas vendues.")

    st.subheader("Quelles sont nos pizzas les plus et les moins vendues ?")
    
    # Regrouper les données par nom de pizza et calculer la somme des quantités
    pizza_sales = df.groupby('pizza_name')['quantity'].sum().reset_index()

    # Trouver la pizza la plus vendue
    most_sold_pizza = pizza_sales[pizza_sales['quantity'] == pizza_sales['quantity'].max()]['pizza_name']

    # Trouver la pizza la moins vendue
    least_sold_pizza = pizza_sales[pizza_sales['quantity'] == pizza_sales['quantity'].min()]['pizza_name']

    # Afficher les résultats
    st.write(f"La pizza la plus vendue est {most_sold_pizza.values[0]} avec {pizza_sales['quantity'].max()} pizzas vendues.")
    st.write(f"La pizza la moins vendue est {least_sold_pizza.values[0]} avec {pizza_sales['quantity'].min()} pizzas vendues.")

    # Regrouper les données par nom et taille de pizza et calculer la somme des quantités
    pizza_sales = df.groupby(['pizza_name', 'pizza_size'])['quantity'].sum().reset_index()

    # Trouver la pizza la plus vendue
    most_sold_pizza = pizza_sales[pizza_sales['quantity'] == pizza_sales['quantity'].max()]

    # Trouver la pizza la moins vendue
    least_sold_pizza = pizza_sales[pizza_sales['quantity'] == pizza_sales['quantity'].min()]

    # Afficher les résultats
    st.write(f"La pizza la plus vendue est {most_sold_pizza['pizza_name'].values[0]} de taille {most_sold_pizza['pizza_size'].values[0]} avec {most_sold_pizza['quantity'].values[0]} pizzas vendues.")
    st.write(f"La pizza la moins vendue est {least_sold_pizza['pizza_name'].values[0]} de taille {least_sold_pizza['pizza_size'].values[0]} avec {least_sold_pizza['quantity'].values[0]} pizzas vendues.")

    # Calculer le revenu total pour chaque taille de pizza
    size_revenue = df.groupby('pizza_size')['total_price'].sum().reset_index()

    # Trouver la taille de pizza qui génère le plus de revenus
    max_size_revenue = size_revenue[size_revenue['total_price'] == size_revenue['total_price'].max()]

    # Trouver la taille de pizza qui génère le moins de revenus
    min_size_revenue = size_revenue[size_revenue['total_price'] == size_revenue['total_price'].min()]

    st.write(f"La taille de pizza qui génère le plus de revenus est {max_size_revenue['pizza_size'].values[0]} avec un revenu total de {max_size_revenue['total_price'].values[0]} USD.")
    st.write(f"La taille de pizza qui génère le moins de revenus est {min_size_revenue['pizza_size'].values[0]} avec un revenu total de {min_size_revenue['total_price'].values[0]} USD.")

    # Calculer le revenu total pour chaque type de pizza
    type_revenue = df.groupby('pizza_name')['total_price'].sum().reset_index()

    # Trouver le type de pizza qui génère le plus de revenus
    max_type_revenue = type_revenue[type_revenue['total_price'] == type_revenue['total_price'].max()]

    # Trouver le type de pizza qui génère le moins de revenus
    min_type_revenue = type_revenue[type_revenue['total_price'] == type_revenue['total_price'].min()]

    st.write(f"Le type de pizza qui génère le plus de revenus est {max_type_revenue['pizza_name'].values[0]} avec un revenu total de {max_type_revenue['total_price'].values[0]} USD.")
    st.write(f"Le type de pizza qui génère le moins de revenus est {min_type_revenue['pizza_name'].values[0]} avec un revenu total de {min_type_revenue['total_price'].values[0]} USD.")

    # Calculer le revenu total pour chaque combinaison de taille et de type de pizza
    combo_revenue = df.groupby(['pizza_name', 'pizza_size'])['total_price'].sum().reset_index()

    # Trouver la combinaison de taille et de type de pizza qui génère le plus de revenus
    max_combo_revenue = combo_revenue[combo_revenue['total_price'] == combo_revenue['total_price'].max()]

    # Trouver la combinaison de taille et de type de pizza qui génère le moins de revenus
    min_combo_revenue = combo_revenue[combo_revenue['total_price'] == combo_revenue['total_price'].min()]

    st.write(f"La combinaison de taille et de type de pizza qui génère le plus de revenus est {max_combo_revenue['pizza_name'].values[0]} (taille {max_combo_revenue['pizza_size'].values[0]}) avec un revenu total de {max_combo_revenue['total_price'].values[0]} USD.")
    st.write(f"La combinaison de taille et de type de pizza qui génère le moins de revenus est {min_combo_revenue['pizza_name'].values[0]} (taille {min_combo_revenue['pizza_size'].values[0]}) avec un revenu total de {min_combo_revenue['total_price'].values[0]} USD.")    

    # Sélectionner les trois pizzas les plus vendues
    top_pizzas = df.groupby('pizza_name')['quantity'].sum().nlargest(3).index

    # Filtrer les données pour ne conserver que ces trois pizzas
    filtered_sales = df[df['pizza_name'].isin(top_pizzas)]

    # Ajouter la colonne 'Month'
    filtered_sales['Month'] = filtered_sales['order_date'].dt.month_name()

    # Créer les graphiques pour les données filtrées
    fig1 = px.bar(filtered_sales, x='Month', y='quantity', color='pizza_name', title='Ventes des trois pizzas les plus commandées par mois')
    fig1.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(fig1)

    fig2 = px.bar(filtered_sales, x='Day_of_week', y='quantity', color='pizza_name', title='Ventes des trois pizzas les plus commandées par jour')
    fig2.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(fig2)

    fig3 = px.bar(filtered_sales, x='Hour', y='quantity', color='pizza_name', title='Ventes des trois pizzas les plus commandées par heure')
    fig3.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })
    st.plotly_chart(fig3)


  

    st.subheader("Quelles sont les ingrédients les plus utilisés ?")

    # Extraire les ingrédients
    ingredients = df['pizza_ingredients']

    # Créer une liste pour stocker tous les ingrédients
    all_ingredients = []

    # Parcourir chaque ligne d'ingrédients
    for row in ingredients:
        # Diviser la chaîne en une liste d'ingrédients
        row_ingredients = row.split(', ')

        # Ajouter chaque ingrédient à la liste de tous les ingrédients
        all_ingredients.extend(row_ingredients)

    # Utiliser Counter pour compter le nombre d'occurrences de chaque ingrédient
    ingredient_counts = Counter(all_ingredients)

    # Afficher les 5 ingrédients les plus courants
    most_common_ingredients = ingredient_counts.most_common(5)

    st.write(most_common_ingredients)

    st.subheader("Analyse des places assisent")

    st.write("""
Dans notre analyse, nous avons utilisé les données de vente de pizzas pour estimer l'utilisation des sièges dans votre restaurant. Nous avons fait plusieurs hypothèses pour réaliser cette estimation :

Nous avons supposé qu'une pizza de taille S, M et L équivaut à une personne assise et qu'une pizza de taille XL, XXL équivaut à deux personnes assises. Cela nous a permis d'estimer le nombre de sièges occupés pour chaque commande en fonction de la taille et de la quantité de pizzas commandées.

Nous avons ensuite regroupé ces estimations par mois, jour et heure pour obtenir une vue détaillée de l'utilisation des sièges dans votre restaurant. Cela nous a permis d'identifier les périodes où votre restaurant est le plus occupé.

Enfin, nous avons calculé le pourcentage d'utilisation des sièges en divisant le nombre total de sièges occupés par la capacité totale du restaurant (60*15=900 sièges).
""")
    # Définir une fonction pour estimer le nombre de sièges occupés en fonction de la taille de la pizza
    def estimate_seats(row):
        if row['pizza_size'] in ['XXL', 'XL']:
            return 2 * row['quantity']
        elif row['pizza_size'] in ['S', 'M', 'L']:
            return row['quantity']
        else:
            return 0

    # Appliquer la fonction à chaque ligne du DataFrame pour estimer le nombre de sièges occupés
    df['estimated_seats'] = df.apply(estimate_seats, axis=1)

    # Calculer le nombre total de sièges occupés
    total_seats = df['estimated_seats'].sum()

    # Calculer le pourcentage d'utilisation des sièges
    percentage = (total_seats / (15 * 60)) * 100

    # Afficher le résultat
    st.write(f"Le pourcentage estimé d'utilisation des sièges est de {round(percentage, 2)}%.")

    
    # Convertir la colonne 'order_time' en datetime
    df['order_time'] = pd.to_datetime(df['order_time'])

    # Extraire l'heure de la colonne 'order_time'
    df['Hour'] = df['order_time'].dt.hour

    # Estimer le nombre de sièges occupés
    df['estimated_seats'] = df.apply(estimate_seats, axis=1)

    # Grouper par heure et calculer la somme des sièges estimés
    hr = df.groupby('Hour')['estimated_seats'].sum().reset_index()

    # Calculer le pourcentage d'utilisation des sièges pour chaque heure
    hr['percentage'] = (hr['estimated_seats'] / 60) * 100

    # Créer une figure pour le pourcentage d'utilisation des sièges
    plt.figure(figsize=(10, 6))

    # Définir les couleurs en fonction du pourcentage d'utilisation des sièges
    colors = np.where(hr['percentage'] < 98, 'g', np.where(hr['percentage'] <= 100, 'orange', 'r'))

    # Tracer un graphique à barres avec les couleurs définies
    plt.bar(hr['Hour'], hr['percentage'], color=colors)

    # Ajouter une ligne horizontale à 100%
    plt.axhline(100, color='black', linestyle='--')

    # Ajouter des titres et des étiquettes d'axe
    plt.xlabel('Heure')
    plt.ylabel('Pourcentage d\'utilisation des sièges')
    plt.title('Pourcentage d\'utilisation des sièges par heure')

    # Créer une légende pour les couleurs
    
    green_patch = mpatches.Patch(color='green', label='Moins de 98%')
    orange_patch = mpatches.Patch(color='orange', label='Entre 98% et 100%')
    red_patch = mpatches.Patch(color='red', label='Plus de 100%')
    plt.legend(handles=[green_patch, orange_patch, red_patch])

    # Afficher le graphique avec Streamlit
    st.pyplot(plt)

    # Calculer le nombre de sièges en trop ou manquants pour chaque heure
    hr['seats_difference'] = hr['estimated_seats'] - 900

    # Créer une liste pour stocker les messages
    messages = []

    # Afficher le nombre de sièges en trop ou manquants pour chaque heure
    for index, row in hr.iterrows():
        hour = row['Hour']
        seats_difference = row['seats_difference']
        percentage = row['percentage']

        if seats_difference > 0:
            messages.append(f"À {hour} heure, il y a {seats_difference} sièges en trop ({percentage}% d'utilisation des sièges).")
        elif seats_difference < 0:
            messages.append(f"À {hour} heure, il manque {-seats_difference} sièges ({percentage}% d'utilisation des sièges).")
        else:
            messages.append(f"À {hour} heure, il n'y a ni sièges en trop ni sièges manquants ({percentage}% d'utilisation des sièges).")

    # Afficher les messages avec Streamlit
    for message in messages:
        st.text(message)


    
    # Regrouper les données par mois, jour et heure et calculer la somme des sièges estimés
    seats_usage = df.groupby(['Month', 'Day_of_week', 'Hour'])['estimated_seats'].sum().reset_index()

    # Trouver le mois, le jour et l'heure avec le plus grand nombre de sièges occupés
    peak_usage = seats_usage[seats_usage['estimated_seats'] == seats_usage['estimated_seats'].max()]

    # Initialiser le DataFrame avec les colonnes appropriées
    peak_usage = pd.DataFrame(columns=['Month', 'Day', 'Hour', 'Seats'])

    # Parcourir chaque combinaison unique de mois, jour et heure
    for month in df['Month'].unique():
        for day in df['Day_of_week'].unique():
            for hour in df['Hour'].unique():
                # Filtrer les données pour le mois, le jour et l'heure spécifiques
                data = df[(df['Month'] == month) & (df['Day_of_week'] == day) & (df['Hour'] == hour)]

                # Calculer le nombre total de sièges occupés
                total_seats = data['estimated_seats'].sum()

                # Si le nombre total de sièges occupés dépasse la capacité du restaurant, ajouter les détails à peak_usage
                if total_seats > 60:
                    peak_usage.loc[len(peak_usage)] = [month, day, hour, total_seats]


    # Regrouper les données par mois et calculer la somme des sièges estimés
    monthly_usage = peak_usage.groupby('Month')['Seats'].sum().reset_index()

    # Créer un graphique à barres pour l'utilisation des sièges par mois
    fig = px.bar(monthly_usage, x='Month', y='Seats', title='Utilisation des sièges par mois')

    # Modifier le fond pour qu'il soit blanc
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # Afficher le graphique avec Streamlit
    st.plotly_chart(fig)

    # Regrouper les données par jour et calculer la somme des sièges estimés
    daily_usage = peak_usage.groupby('Day')['Seats'].sum().reset_index()

    # Créer un graphique à barres pour l'utilisation des sièges par jour
    fig = px.bar(daily_usage, x='Day', y='Seats', title='Utilisation des sièges par jour de la semaine')

    # Modifier le fond pour qu'il soit blanc
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # Afficher le graphique avec Streamlit
    st.plotly_chart(fig)

    # Regrouper les données par heure et calculer la somme des sièges estimés
    hourly_usage = peak_usage.groupby('Hour')['Seats'].sum().reset_index()

    # Créer un graphique à barres pour l'utilisation des sièges par heure
    fig = px.bar(hourly_usage, x='Hour', y='Seats', title='Utilisation des sièges par heure')

    # Ajouter une ligne horizontale à 900
    fig.add_shape(
        type="line",
        xref="paper", yref="y",
        x0=0, y0=900,
        x1=1, y1=900,
        line=dict(
            color="Black",
            width=3,
            dash="dashdot",
        )
    )

    # Modifier le fond pour qu'il soit blanc
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # Afficher le graphique avec Streamlit
    st.plotly_chart(fig)


elif selection == "Conclusion":
    st.header("Conclusion")
    st.write("""
    Suite à une analyse approfondie des données de vente de notre restaurant, voici quelques observations clés qui pourraient nous aider à améliorer nos opérations et notre rentabilité :

    Ventes de pizzas : Nous avons observé un pic notable de ventes de pizzas en juillet, suivi de très faibles ventes en septembre, octobre et novembre. Cela pourrait indiquer une opportunité d'augmenter nos efforts de marketing pendant ces mois plus lents pour stimuler les ventes.

    Jours et heures de pointe : Le vendredi est le jour où nous avons le plus de clients et le midi est le moment où nous avons le plus de clients. Pendant cette période de pointe, nous vendons environ 14% de nos pizzas. Cela pourrait nous aider à planifier notre personnel et nos ressources de manière plus efficace.

    Pizzas les plus populaires : Nos pizzas les plus vendues sont "The Classic Deluxe Pizza", "The Barbecue Chicken Pizza" et "The Hawaiian Pizza". Nous pourrions envisager de mettre en avant ces pizzas dans nos menus et nos promotions.

    Utilisation des sièges : Nous avons estimé que l'utilisation des sièges était la plus élevée en mars, avec un total de 2387 sièges utilisés. Le vendredi est le jour où nous utilisons le plus de sièges et l'heure de pointe pour l'utilisation des sièges est à midi. Ces informations pourraient nous aider à gérer efficacement notre capacité d'accueil.

    Ces observations sont basées sur plusieurs hypothèses, notamment que chaque pizza vendue équivaut à un siège occupé (deux pour les tailles XL et XXL) et que notre restaurant a une capacité totale de 60 sièges. Bien que ces hypothèses puissent ne pas être parfaitement précises, elles nous donnent une bonne estimation de l'utilisation des sièges dans notre restaurant.

    En se basant sur ces informations, je recommanderais d'envisager les actions suivantes :

    Augmenter nos efforts de marketing pendant les mois où les ventes sont faibles.
    Planifier notre personnel en fonction des jours et des heures de pointe.
    Mettre en avant nos pizzas les plus populaires dans nos menus et nos promotions.
    
    Bien sûr, je m'excuse pour l'omission. Voici quelques observations supplémentaires concernant les revenus générés par nos ventes de pizzas :

    Revenus par taille de pizza : La taille de pizza qui génère le plus de revenus est la taille L avec un revenu total de 375318,7 USD. À l'inverse, la taille XXL génère le moins de revenus avec un total de seulement 1006,60 USD. Cela pourrait indiquer que nos clients préfèrent les pizzas de taille moyenne plutôt que les plus grandes.

    Revenus par type de pizza : Le type de pizza qui génère le plus de revenus est "The Thai Chicken Pizza" avec un revenu total de 43434,25 USD. Le moins rentable est "The Brie Carre Pizza" avec un revenu total de 11588,5 USD. Cela pourrait nous aider à décider quelles pizzas promouvoir davantage.

    Revenus par combinaison de taille et de type : La combinaison qui génère le plus de revenus est "The Thai Chicken Pizza" (taille L) avec un revenu total de 29257,5 USD. La combinaison qui génère le moins de revenus est "The Greek Pizza" (taille XXL) avec un revenu total de seulement 1006,60 USD.

    Sur la base de ces observations, je recommanderais d'envisager les actions suivantes :

    Promouvoir davantage nos pizzas de taille L, qui semblent être les plus populaires auprès de nos clients.
    Mettre en avant "The Thai Chicken Pizza" dans nos menus et nos promotions, car c'est le type de pizza qui génère le plus de revenus.
    Réévaluer notre offre pour les pizzas XXL et "The Greek Pizza", qui semblent moins populaires et moins rentables.
    
    Nous avons aussi analysé le prix des pizzas de taille L en fonctions des ingrédients a l'interieur t nous observons que les tomates ont une corrélation positive élevée de 0,626242 avec le prix, ce qui signifie que les pizzas contenant des tomates ont tendance à être plus chères. À l’inverse, le fromage mozzarella a une corré
    """)

using Il2Cpp;
using MelonLoader;
using System;
using System.Collections;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine.Events;
using UnityEngine;
using UnityEngine.Playables;
using Microsoft.VisualBasic;
using UnityEngine.UI;

namespace MitaAI
{
    public static class UINeuroMita
    {
        static GameObject MenuObject;
        static Menu MainMenu;
        static GameObject StartMenuObject;
        static MenuLocation StartMenu;

        static MenuLocation MainMenuLocation;
        static GameObject FrameMenuObject;

        static GameObject NeuroMitaButton;

        public static void init()
        {
            MenuObject = GameObject.Find("MenuGame");
            MainMenu = MenuObject.GetComponent<Menu>();
            FrameMenuObject = MenuObject.transform.Find("Canvas/FrameMenu").gameObject;
            //makeButtonTemplate();

            ButtonNeuroMita();

            MenuNeuroMita();

            

        }


        static void makeButtonTemplate()
        {

        }

        public static void ButtonNeuroMita()
        {
            #region ButtonLoading

            MelonLogger.Msg("Start SceneMenu");
            // Кнопки мода
            GameObject Menu = GameObject.Find("MenuGame/Canvas/FrameMenu/Location Menu").gameObject;
            MainMenuLocation = Menu.GetComponent<MenuLocation>();
            Menu.transform.localPosition = new Vector3(250f, 355f, 0);
            Menu.transform.Find("Text").localPosition = new Vector3(-250, 15, 0);

            NeuroMitaButton = GameObject.Instantiate(Menu.transform.Find("Button Continue").gameObject);
            MainMenuLocation.objects.Add(NeuroMitaButton.GetComponent<RectTransform>());
            NeuroMitaButton.name = "NeuroMitaStartButton";


            NeuroMitaButton.transform.SetParent(GameObject.Find("MenuGame/Canvas/FrameMenu/Location Menu").transform);
            NeuroMitaButton.transform.localPosition = new Vector3(-250, -45, 0);
            NeuroMitaButton.transform.localScale = new Vector3(1, 1, 1);
            NeuroMitaButton.transform.rotation = new Quaternion(0, 0, 0, 0);


            GameObject NeuroMitaButtonText = NeuroMitaButton.transform.Find("Text").gameObject;
            //NeuroMitaButtonText.GetComponent<Localization_UIText>().deactiveTextTranslate = true;
            //NeuroMitaButtonText.GetComponent<Localization_UIText>().enabled = false;
            //NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().text = "ИГРАТЬ С NEUROMITA";
            //NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().m_Text = "ИГРАТЬ С NEUROMITA";

            MelonCoroutines.Start(changeName(NeuroMitaButtonText, "ИГРАТЬ С NEUROMITA"));

            //NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().font  = Menu.transform.Find("Button NewGame/Text").GetComponent<Text>().font;


            UI_Colors uI_Colors = NeuroMitaButton.GetComponent<UI_Colors>();

            // Цвет фона
            uI_Colors.SetColorImage(0, new Color(0.5f, 1f, 0.5f, 0.5f));

            MitaCore.Instance.sendSystemInfo("Игрок в меню");
            

            try
            {
                ButtonMouseClick buttonMouseClick = NeuroMitaButton.GetComponent<ButtonMouseClick>();
                buttonMouseClick.eventClick = setupMenuEvent(buttonMouseClick.gameObject, "ButtonLoad");
            }
            catch (Exception e)
            {

                MelonLogger.Error(e);
            }



            #endregion
        }
        private static GameObject makeButton()
        {
            return null;
        }



        public static void MenuNeuroMita()
        {
            MelonLogger.Msg(1);

            StartMenuObject = GameObject.Instantiate(FrameMenuObject.transform.Find("Location MainOptions").gameObject, FrameMenuObject.transform);
            StartMenuObject.active = false;
            StartMenu = StartMenuObject.GetComponent<MenuLocation>();
            MelonLogger.Msg(2);
            try
            {
                MelonCoroutines.Start( changeName(StartMenuObject.transform.Find("Text").gameObject, "Настройки NeuroMita") );
            }
            catch (Exception e) { MelonLogger.Error(e); }
            MelonLogger.Msg(3);

            // Button Full Clear
            // Button Option Graphics
            // Button Option Game

            try
            {
                GameObject MitaCrazyButton = StartMenuObject.transform.Find("Button Full Clear").gameObject;
                MitaCrazyButton.active = false;
                MitaCrazyButton = GameObject.Instantiate(NeuroMitaButton, MitaCrazyButton.transform.position, MitaCrazyButton.transform.rotation, MitaCrazyButton.transform.parent);
                MitaCrazyButton.name = "MitaCrazyButton";
                MitaCrazyButton.active = true;

                MelonCoroutines.Start(changeName(MitaCrazyButton.transform.Find("Text").gameObject, "Безумная Мита"));
                MitaCrazyButton.GetComponent<ButtonMouseClick>().eventClick = setupMenuEvent(MitaCrazyButton, MitaCrazyButton.name);

            }
            catch (Exception e) {MelonLogger.Error(e); }
            MelonLogger.Msg(4);
            try
            {
                GameObject MitaKindButton = StartMenuObject.transform.Find("Button Option Graphics").gameObject;
                MitaKindButton.active = false;
                MitaKindButton = GameObject.Instantiate(NeuroMitaButton, MitaKindButton.transform.position, MitaKindButton.transform.rotation, MitaKindButton.transform.parent);
                MitaKindButton.name = "MitaKindButton";
                MitaKindButton.active = true;

                MelonCoroutines.Start(changeName(MitaKindButton.transform.Find("Text").gameObject, "Добрая Мита"));
                MitaKindButton.GetComponent<ButtonMouseClick>().eventClick = setupMenuEvent(MitaKindButton, MitaKindButton.name);

            }
            catch (Exception e) { MelonLogger.Error(e); }

            try
            {
                GameObject MitaShorthairButton = StartMenuObject.transform.Find("Button Option Game").gameObject;
                MitaShorthairButton.active = false;
                MitaShorthairButton = GameObject.Instantiate(NeuroMitaButton, MitaShorthairButton.transform.position, MitaShorthairButton.transform.rotation, MitaShorthairButton.transform.parent);
                MitaShorthairButton.name = "MitaShortButton";
                MitaShorthairButton.active = true;

                MelonCoroutines.Start(changeName(MitaShorthairButton.transform.Find("Text").gameObject, "Коротковолосая Мита"));
                MitaShorthairButton.GetComponent<ButtonMouseClick>().eventClick = setupMenuEvent(MitaShorthairButton, MitaShorthairButton.name);

            }
            catch (Exception e) { MelonLogger.Error(e); }

            try
            {
                MelonLogger.Msg("Cappy");
                GameObject MitaCappyButton = StartMenuObject.transform.Find("Button Volume").gameObject;
                MitaCappyButton.active = false;
                MitaCappyButton = GameObject.Instantiate(NeuroMitaButton, MitaCappyButton.transform.position, MitaCappyButton.transform.rotation, MitaCappyButton.transform.parent);
                MitaCappyButton.name = "MitaCappyButton";
                MitaCappyButton.active = true;

                MelonCoroutines.Start(changeName(MitaCappyButton.transform.Find("Text").gameObject, "Кепочка"));
                MitaCappyButton.GetComponent<ButtonMouseClick>().eventClick = setupMenuEvent(MitaCappyButton, MitaCappyButton.name);

            }
            catch (Exception e) { MelonLogger.Error(e); }


            //GameObject MitaShortButton = StartMenuObject.transform.Find("Button Volume").gameObject;
            //MitaShortButton.active = false;
            MelonLogger.Msg(5);
            //GameObject MitaShortButton = StartMenuObject.transform.Find("Back").gameObject;
            //MitaShortButton.active = false;



            GameObject Back = StartMenuObject.transform.Find("Button Back").gameObject;
            Back.name = "ButtonReturn";
            Back.GetComponent<ButtonMouseClick>().eventClick = setupMenuEvent(Back, Back.name);

            MelonLogger.Msg(6);
        }
        static public void MenuEventsCases(string eventName){
            MelonLogger.Msg($"MenuEventsCases: {eventName}");

            eventName = eventName.Substring(MenuPrefix.Length);
            switch (eventName)
            {
                case "ButtonLoad":
                    StartMenu.Active(true); 
                    //MitaCore.MainMenu.ButtonLoadScene(MitaCore.Instance.requiredSave);
                    break;
                case "MitaShortButton":
                    Settings.MitaType.Value = MitaCore.character.ShortHair;
                    Settings.Save();
                    MainMenu.ButtonLoadScene(MitaCore.Instance.requiredSave);
                    break;
                case "MitaCrazyButton":
                    Settings.MitaType.Value = MitaCore.character.Mita;
                    Settings.Save();
                    MainMenu.ButtonLoadScene(MitaCore.Instance.requiredSave);
                    break;
                case "MitaKindButton":
                    Settings.MitaType.Value = MitaCore.character.Kind;
                    Settings.Save();
                    MainMenu.ButtonLoadScene(MitaCore.Instance.requiredSave);
                    break;
                case "MitaCappyButton":
                    Settings.MitaType.Value = MitaCore.character.Cappy;
                    Settings.Save();
                    MainMenu.ButtonLoadScene(MitaCore.Instance.requiredSave);
                    break;
                case "ButtonReturn":
                    StartMenu.Active(false);
                    MainMenuLocation.Active(true);
                    break;
                default:
                    MelonLogger.Warning($"Unhandled button {eventName}");
                    break;
            }
        }

        static IEnumerator changeName(GameObject NeuroMitaButtonText,string text)
        {
            Font original_font = NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().font;
            NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().text = text;
            NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().m_Text = text;

            float time = 0f;
            while (NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().font == original_font)
            {
                if (time >= 30f)
                {
                    break;
                }

                time += Time.unscaledDeltaTime;
                yield return null;
            }
            NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().text = text;
            NeuroMitaButtonText.GetComponent<UnityEngine.UI.Text>().m_Text = text;
        }

        public static string MenuPrefix = "MENU|";

        static private UnityEvent setupMenuEvent(GameObject gameObject, string eventName)
        {
            EventsProxy eventsProxy = gameObject.AddComponent<EventsProxy>();
            return eventsProxy.SetupEvent($"{MenuPrefix}{eventName}");
        }
    }
}

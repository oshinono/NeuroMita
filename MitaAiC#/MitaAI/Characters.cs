using MelonLoader;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MitaAI
{
    public static class Characters
    {

        // Сюда перенести все сharacter и changeMita
        static MitaCore.character cart = MitaCore.character.None;

        public static MitaCore.character get_cart()
        {
            if (cart == MitaCore.character.None) init_cart();
            return cart;
        }

        private static void init_cart()
        {
            if (Utils.Random(1, 2)) cart = MitaCore.character.Cart_portal;
            else cart = MitaCore.character.Cart_divan;
        }

        public static MitaCore.character ChooseCharacterToAsnwer()
        {
            if ( Utils.getDistanceBetweenObjects(MitaCore.Instance.playerObject, MitaCore.Instance.cartridgeReader)<3f && MitaCore.Instance.getDistanceToPlayer()>3f){

                MelonLogger.Msg("Sent to cart");        
                return get_cart();
            }

            return MitaCore.Instance.currentCharacter;
        }


    }
}

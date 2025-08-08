import { initializeApp } from "firebase/app";
import { getMessaging, getToken, onMessage } from "firebase/messaging";

const firebaseConfig = {
  apiKey: "AIzaSyBVZuGwmpcEXa8M10f4zeKIrHm01P5kGRw",

  authDomain: "personal-agent-ste.firebaseapp.com",

  projectId: "personal-agent-ste",

  storageBucket: "personal-agent-ste.firebasestorage.app",

  messagingSenderId: "394449515613",

  appId: "1:394449515613:web:5b078c6b9f47370818c9b4"

};

const app = initializeApp(firebaseConfig);
const messaging = getMessaging(app);

export const requestPermission = async () => {
  try {
    const token = await getToken(messaging, { vapidKey: "BEvz_DrfgLcvBuJgwSaY2imvz2QJ9fx5qLPJJ-ZW483lwYbLBdBUnfisj2-Sw7XzcuwmvNP8Ljk-I5Y17hqI_sg" });
    console.log("Push token:", token);
    return token;
  } catch (err) {
    console.error("Permission denied or error", err);
  }
};

export const listenForMessages = () => {
  onMessage(messaging, (payload) => {
    console.log("Push message received", payload);
  });
};
